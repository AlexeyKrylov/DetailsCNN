import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        model = []
        # 32
        model += [nn.Conv2d(3, 6, 5, stride=1, padding=0)]
        model += [nn.ReLU()]
        model += [nn.MaxPool2d((2, 2))]
        # 16
        model += [nn.Conv2d(6, 16, 5, stride=1, padding=0)]
        model += [nn.ReLU()]
        model += [nn.MaxPool2d((2, 2))]
        # 8
        model += [nn.Conv2d(16, 1, 5, stride=1, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input).reshape(-1)
