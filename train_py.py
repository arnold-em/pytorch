import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model import *

train_datasets = torchvision.datasets.CIFAR10('./Pytorch',transform=torchvision.transforms.ToTensor())
test_datasets = torchvision.datasets.CIFAR10('./Pytorch',train=False,transform=torchvision.transforms.ToTensor())

trains_load = DataLoader(train_datasets,batch_size=64)
test_load = DataLoader(test_datasets,batch_size=64)


writer = SummaryWriter('train_py')
epoch = 4
train_step = 0
test_setp = 0
lossf = nn.CrossEntropyLoss()

model = Model()
optim = torch.optim.SGD(model.parameters(),lr=0.01)
for i in range(epoch):
    print('the {} epoch:'.format(i+1))
    total_accuracy = 0
    total_loss = 0
    for trains in trains_load:
        imgs,labels = trains
        result = model(imgs)
        loss = lossf(result,labels)
        total_loss += loss
        accury = (result.argmax(1) == labels)
        total_accuracy += list(accury).count(True)
        optim.zero_grad()
        loss.backward()
        writer.add_scalar('train',loss.item(),train_step)
        optim.step()
        if train_step % 100 == 0:
            print('step{},loss:{}'.format(train_step,total_loss))
        train_step += 1
    print('the accurate:{}'.format(total_accuracy/len(train_datasets)))
    if i % 2 == 0:
        torch.save(model.state_dict(),'model.pth{}'.format(i+1))
        print('模型保存成功')

total_test_loss = 0
total_test_acc = 0
test_step = 0
with torch.no_grad():
    for test in test_load:
        
        imgs,labels = test
        result = model(imgs)
        loss = lossf(result,imgs)
        total_test_loss += loss
        accury = (result.argmax(1)==labels)
        total_test_acc += list(accury).count(True)
        writer.add_scalar('test_loss',loss.item(),test_step)
        writer.add_scalar('test_acc',total_test_acc,test_step)

print('accurate:'.format(total_test_acc/len(test_load)))





