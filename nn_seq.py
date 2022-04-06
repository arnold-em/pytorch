import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # self.conv1 = nn.Conv2d(3,32,5,padding=2)
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32,32,5,padding=2)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32,64,5,padding=2)
        # self.pool3 = nn.MaxPool2d(2)
        # self.fallten = nn.Flatten()
        # self.linear1 = nn.Linear(1024,64)
        # self.linear2 = nn.Linear(64,10)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.fallten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model(x)
        return x



x = torch.ones((64,3,32,32))

print(x)

t = Model()
print(t)
print(t(x))

writer = SummaryWriter('log_nn_seq')
writer.add_graph(t,x)
writer.close()