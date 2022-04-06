import torch
import torch.nn as nn

a = torch.tensor((1,2,5),dtype=torch.float32)
b = torch.tensor((1,2,3),dtype=torch.float32)
print(a)
print(a.shape)
c = torch.reshape(a,(1,1,1,3))
d = torch.reshape(b,(1,1,1,3))
print(c)
print(c.shape)
L11 = nn.L1Loss()

L2 = nn.MSELoss(reduction='mean')
L22 = nn.MSELoss(reduction='sum')
result1 = L11(a,b)
result2 = L11(c,d)
print(result1)
print(result2)

result3 = L2(a,b)
result4 = L22(a,b)

print(result3,result4)
L3 = nn.CrossEntropyLoss()
a = torch.tensor([0.1,0.1,0.8],dtype=torch.float32)
e = torch.tensor([1])
print(e)

a = torch.reshape(a,(1,3))
result5 = L3(a,e)

print(result5)
