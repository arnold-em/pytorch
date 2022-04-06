from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter('logs')

img_path = 'train/ants_image/0013035.jpg'

img = Image.open(img_path)

img = np.array(img)

writer.add_image('pic',img,dataformats='HWC')
writer.close() 