from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'images/8398478_50ef10c47a.jpg'

img = Image.open(img_path)
totensor = transforms.ToTensor()
img_tensor = totensor(img)
normal = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_data = normal(img_tensor)



resize_trans = transforms.Resize((512,512))
resize_trans2 = transforms.Resize(512)
com_trans = transforms.Compose([resize_trans2,totensor])

img2 = com_trans(img)

writer = SummaryWriter('logs')




randomcrop = transforms.RandomCrop((200,200))

for i in range(10):
    imgs = randomcrop(img)
    imgs = totensor(imgs)
    writer.add_image('random',imgs,i)



writer.close()