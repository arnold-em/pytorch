import torchvision
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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

    def forward(self, x):
        return self.model(x)


trans = torchvision.transforms.Compose(
    transforms=[torchvision.transforms.ToTensor()]
)
dataset = torchvision.datasets.CIFAR10('./Pytorch', train=False, transform=trans)

data = DataLoader(dataset, batch_size=4, shuffle=True)
loss = nn.CrossEntropyLoss()

model = Model()
optim = torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(5):
    total_loss = 0.0
    for item in data:
        imgs, labels = item
        loss_result = model(imgs)
        result = loss(loss_result, labels)
        optim.zero_grad()
        result.backward()
        optim.step()
        total_loss += result

    print('numer %2d epoch loss:%.5f'%(epoch+1,total_loss))
