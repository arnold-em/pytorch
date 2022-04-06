import torchvision
from torch.utils.tensorboard import SummaryWriter

trans_compose = torchvision.transforms.Compose(
    transforms=[torchvision.transforms.ToTensor()]
)

train = torchvision.datasets.CIFAR10('./Pytorch',train=True,transform=trans_compose,download=True)
test = torchvision.datasets.CIFAR10('./Pytorch',train=False,transform=trans_compose,download=True)

print(train)


writer = SummaryWriter('cifar10')

for i in range(10):
    img,label = test[i]
    writer.add_image('test',img,i)

writer.close()
