import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)

        )

    def forward(self,x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    models = Model()
    x = torch.ones((64,3,32,32))
    z = models(x)
    print(z.shape)

    t = torch.ones((64,3,32,32))
    print(t.shape)
