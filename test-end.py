import torchvision
from PIL import Image

from Model import *

model = Model()
model.load_state_dict(torch.load('train_data.pth'))
img_path = 'image/dog.png'
img = Image.open(img_path)
d = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
print(img)

trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                torchvision.transforms.ToTensor()])

img = trans(img)
img = torch.reshape(img,(1,3,32,32))

pre = model(img)
t = pre.argmax(1)

for k,v in d.items():
    if d[k] == t.item():
        print('pre:',k)
