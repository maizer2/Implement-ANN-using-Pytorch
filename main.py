from train.CNN.AlexNet.train_AlexNet import train_AlexNet
from train.CNN.LeNet5.train_LeNet5 import train_LeNet5
from train.CNN.VGG.train_VGG import train_VGG

# ----------------------------------------------

train_VGG(img_channels=1, vgg_layers=19)
# train_AlexNet()
# train_LeNet5()