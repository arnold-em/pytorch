from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg'
img_pil = Image.open(img_path)
totensor = transforms.ToTensor()

img_tensor = totensor(img_pil)
print(img_pil)
print(img_tensor)

writer = SummaryWriter('tag')

writer.add_image('tensor_img',img_tensor)
writer.close()