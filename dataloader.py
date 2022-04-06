import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.Compose(
    transforms=[torchvision.transforms.ToTensor()]
)
writer = SummaryWriter('testpic')
dataset = torchvision.datasets.CIFAR10('./Pytorch',train=False,transform=transform)
dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,
                        drop_last=False,num_workers=0)


for i in range(2):
    step = 1
    for data in dataloader:
        imgs,labels = data

        writer.add_images('test{}'.format(i),imgs,step)
        step += 1

writer.close()