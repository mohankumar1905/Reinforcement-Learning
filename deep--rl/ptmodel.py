import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

class ANN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4, 4000)
        self.hidden2 = nn.Linear(4000, 1)
        lr = 10e-5
        self.optimizer = optim.SGD(self.parameters(), lr)
        self.loss_criterion = nn.MSELoss()
    
    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        return x

    def partial_fit(self, input, target):
        self.optimizer.zero_grad()
        input = Variable(torch.from_numpy(input).float(), requires_grad=False)
        target = Variable(torch.from_numpy(target).float(), requires_grad=False)
        predicted = self(input)
        loss = self.loss_criterion(target, predicted)
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        x = Variable(torch.from_numpy(x).float(), requires_grad=False)
        with torch.no_grad():
            predicted = self(x)
        return predicted