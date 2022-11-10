from train.CNN.AlexNet.train_AlexNet import train_AlexNet
from train.CNN.LeNet5.train_LeNet5 import train_LeNet5
from train.CNN.VGG.train_VGG import train_VGG
from train.CNN.ResNet.train_ResNet import train_ResNet

# ----------------------------------------------

# train_LeNet5()
# train_AlexNet()
# train_VGG(batch_size=575, img_channels=1, layers=11, use_gpu=1)
# train_VGG(batch_size=360, img_channels=1, layers=13, use_gpu=1)
# train_VGG(batch_size=320, img_channels=1, layers=16, use_gpu=0)
# train_VGG(batch_size=280, img_channels=1, layers=19, use_gpu=0)
# train_ResNet(batch_size=2050, img_channels=1, layers=18, use_gpu=0)
# train_ResNet(batch_size=1300, img_channels=1, layers=34, use_gpu=0)
# train_ResNet(batch_size=500, img_channels=1, layers=50, use_gpu=0)
# train_ResNet(batch_size=350, img_channels=1, layers=101, use_gpu=0)
train_ResNet(batch_size=250, img_channels=1, layers=152, use_gpu=0)