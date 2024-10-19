from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    """
    # TODO: Implement for Task 2.1.
    convert = 0
    for pos in range(len(index)): # ex: index = (2, 3), strides = (3, 1) -> (1, 1) = position 4
        convert += index[pos] * strides[pos]
    return convert


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # storage list position to shape position, ex: ordinal = 5 (in storage), calculate the out_index
    def calc_stride() -> Sequence[int]:
        stride = []
        for x in range(len(shape)):
            track = 1
            for i in range(x + 1, len(shape)):
                track *= shape[i]
                # chatgpt gave me this as a way to get the stride of a tensor by using the shape:
                # the stride is the product of all following dimensions
            stride.append(track)
        return stride

    stride = calc_stride()
    for x in range(len(shape)):
        out_index[x] = ordinal // stride[x]
        ordinal = ordinal % stride[x]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None

    """
    # TODO: Implement for Task 2.2.
    # chatgpt helped and I did examples to see what this was doing
    big_ndim = len(big_shape)
    small_ndim = len(shape)
    track = []
    for i in range(max(big_ndim, small_ndim)):
        big_dim_index = big_ndim - 1 - i  # Index from the end of big_shape
        small_dim_index = small_ndim - 1 - i  # Index from the end of shape (going backwards to handle broadcasting)
        
        if big_dim_index >= 0: # check this so the indexing would be valid
            big_idx = big_index[big_dim_index]
        else:
            big_idx = 0
        
        if small_dim_index >= 0:  # If we're within bounds of the smaller shape
            # Use the index for the small tensor, adjusting as necessary
            if shape[small_dim_index] == big_shape[big_dim_index]:
                # If dimensions match, keep the index (since at least 1 dimension stays consistent between the big index and the small index)
                track.append(big_idx)
            elif shape[small_dim_index] == 1:
                # if the small dimension is 1, this means it got broadcasted to the bigger tensor
                track.append(0) # Typically broadcasted indices default to 0
            else:
                raise ValueError("Incompatible dimensions for broadcasting.")
        else:
            # If small_dim_index is out of bounds, we can disregard
            continue

    track.reverse() # since it was added on backwards
    out_index = np.array(track)


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.
    newShape = []
    pad_ones = shape2
    compare = shape1
    if len(shape1) != len(shape2):
        pad_ones = []
        for x in range(abs(len(shape1) - len(shape2))):
            pad_ones.append(1)
        match (len(shape1) > len(shape2)):
            case 1:
                pad_ones += shape2
                compare = shape1
            case 0:
                pad_ones += shape1
                compare = shape2

    if tuple(compare) == tuple(pad_ones):
        return tuple(pad_ones)

    for dim_index in range(len(compare)):
        if (compare[dim_index] * pad_ones[dim_index] != compare[dim_index]) and (compare[dim_index] * pad_ones[dim_index] != pad_ones[dim_index]) and (compare[dim_index] != pad_ones[dim_index]):
            # the current index of the shape should be 1 * something in order to make it broadcastable, incorporates if the 2 are the same, since that should just propogate to the new shape
            raise IndexingError
        else:
            newShape.append(pad_ones[dim_index]) if (compare[dim_index] * pad_ones[dim_index] == pad_ones[dim_index]) else newShape.append(compare[dim_index])
    return tuple(newShape)



def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Get iterable of indices"""
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get data stored at certain index"""
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set data at certain index to given value"""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        newShape = []
        newStride = []
        for a in order:
            newShape.append(self.shape[a])
            newStride.append(self._strides[a]) # self._strides gives numpy integers for some reason.
        x = TensorData(storage = self._storage, shape = tuple(newShape), strides = tuple(newStride)) # need to change stride order and shape
        return x

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
