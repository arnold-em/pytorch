import torch
import torchvision
print(torch.cuda.is_available())
print(torch.__version__)
print(torchvision.__version__)

t = [True,False,False,True]
ht = [True,False,False,True]

print((t==ht).sum())