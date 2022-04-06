import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Cht(nn.Module):

    def __init__(self) -> None:
        super(Cht, self).__init__()
        self.conv = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2,ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)

        return self.pool(x)


trans = torchvision.transforms.Compose(transforms=[
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10('./Pytorch', train=False,
                                         transform=trans)

data = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

writer = SummaryWriter('xxxpic')
model = Cht()
for i in range(2):
    step = 1
    for item in data:

        imgs, label = item
        if i == 1:
            imgs = model(imgs)

        writer.add_images('test{}'.format(i), imgs, step)
        step += 1
