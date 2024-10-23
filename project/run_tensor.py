"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
from minitorch.tensor_data import Shape


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


# TODO: Implement for Task 2.5.


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()  # gets all of the Module abilities
        self.layer1 = Linear(
            2, hidden_layers
        )  # input is 1, output is # of hidden layers
        self.layer2 = Linear(
            hidden_layers, hidden_layers
        )  # input is same as prev layer output
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])  # append an empty list to the weights list
            for j in range(out_size):
                self.weights[
                    i
                ].append(  # add things into the empty list placed in the list of weights
                    self.add_parameter(f"weight_{i}_{j}", RParam(in_size, out_size))
                )
        for j in range(out_size):
            self.bias.append(self.add_parameter(f"bias_{j}", RParam(1, out_size)))

    def forward(self, inputs):
        # adjusted_inputs = []
        # y = [b.value for b in self.bias]
        # for i, x in enumerate(inputs):
        #     for j in range(len(y)):
        #         y[j] = y[j] + x * self.weights[i][j].value
        return y


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
