from train.CNN.AlexNet.train_AlexNet import train_AlexNet
from train.CNN.LeNet5.train_LeNet5_mulGPU import train_LeNet5
from train.CNN.VGG.train_VGG import train_VGG
from train.CNN.ResNet.train_ResNet import train_ResNet
from train.AE.train_AE import train_AE
from train.VAE.vanilaVAE.train_vanilaVAE import train_vanilaVAE
from train.GAN.VanilaGAN.train_VanilaGAN import train_VanilaGAN

# ----------------------------------------------

# train_LeNet5()
# train_AlexNet()
# train_VGG(batch_size=32, img_channels=1, layers=11, use_gpu=1)
# train_VGG(batch_size=32, img_channels=1, layers=13, use_gpu=1)
# train_VGG(batch_size=32, img_channels=1, layers=16, use_gpu=0)
# train_VGG(batch_size=32, img_channels=1, layers=19, use_gpu=0)
# train_ResNet(batch_size=32, img_channels=1, layers=18, use_gpu=0)
# train_ResNet(batch_size=32, img_channels=1, layers=34, use_gpu=0)
# train_ResNet(batch_size=32, img_channels=1, layers=50, use_gpu=0)
# train_ResNet(batch_size=32, img_channels=1, layers=101, use_gpu=1)
# train_ResNet(batch_size=32, img_channels=1, layers=152, use_gpu=2)
# train_AE(batch_size=2048,num_epochs=1000, use_gpu=0, img_channels=1)
# train_vanilaVAE(img_channels=1)
train_VanilaGAN()