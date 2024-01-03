from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        
        #Fully connected layer
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        #Activation
        self.relu = nn.ReLU()

        #Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
