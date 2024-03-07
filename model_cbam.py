import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, out, stride=1,kernel_size=1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(out, in_planes, stride=1,kernel_size=1 ,bias=False))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(planes*4, planes, kernel_size=1, bias=False)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out=self.conv4(out)
        #print(out.shape,'1')
        out = self.ca(out) * out
        out = self.sa(out) * out
        #print(out.shape,'2')
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out
  
    

class ResNet(nn.Module):

    def __init__(self,  layers=5,block=BasicBlock,):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2,
        #                        bias=False)
        self.made_layer_1(inplanes=64,planes=64,block=BasicBlock,num_blocks=20)
        
        self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 32, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5, inplace=False),
                )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(32, 16, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5, inplace=False),
                )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(16, 1, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(64),
                    #nn.LeakyReLU(0.2, inplace=True),
                    #nn.Dropout(0.5, inplace=False),
                )
        self.conv6 = nn.Sequential(
                        nn.Linear(243, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )
        self.conv7 = nn.Sequential(
                    nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5, inplace=False),
                    nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.5, inplace=False),
                    nn.Conv2d(16, 10, 3, 1, 1, bias=False),
                    nn.Conv2d(10,10,5,1,1),
                    nn.Conv2d(10,10,5,1,1),
                    nn.BatchNorm2d(10),
                    nn.AdaptiveAvgPool2d(1)
                )

     
    def _make_layer(self, block, inplanes,planes, num_blocks):#由于残差链接，输入应当与输出相同
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    
    def forward(self,x):
        x=self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#到此为止是64通道
        x=self.made_layer_1(x)#到此为止仍64
        x1=self.conv3(x)
        x1=self.conv4(x1)
        x1=self.conv5(x1)
        x1=x1.reshape(1,-1)
        #print(x1.shape)
        x1=self.conv6(x1)#x1是概率
        x1=x1.squeeze()
        x1=self.sigmoid(x1)
        x2= self.conv7(x)
        x2=x2.squeeze().unsqueeze(0).squeeze()
        return x1,x2
        #return patches_flattened
    

    

class _netG_CIFAR10(nn.Module):
    def __init__(self,  nz):
        super(_netG_CIFAR10, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False
                               
                               ),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output