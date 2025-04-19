import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        #padding=1保证大小不变
        self.conv1 = nn.Conv2d(input_features, output_features, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(output_features)
        self.conv2 = nn.Conv2d(output_features, output_features, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(output_features)
        #初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.ones_(self.bn1.weight)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.bn1.bias)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class UpSample(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        #大小×2,上采样
        self.conv1 = nn.ConvTranspose2d(input_features, output_features, 2, 2)
        #初始化
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        x = self.conv1(x)
        return x
        

#假设输入为3通道的图像(偶数)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        #网络层定义
        #4次下采样 / 2
        #1次增加通道，下采样
        #4次上采样
        self.conv1 = Conv(3, 64)
        self.conv2 = Conv(64, 128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.conv5 = Conv(512, 1024)
        self.upsample1 = UpSample(1024, 512)
        self.conv6 = Conv(1024, 512)
        self.upsample2 = UpSample(512, 256)
        self.conv7 = Conv(512, 256)
        self.upsample3 = UpSample(256, 128)
        self.conv8 = Conv(256, 128)
        self.upsample4 = UpSample(128, 64)
        self.conv9 = Conv(128, 64)
        #2通道表示正类和负类
        self.conv1x1 = nn.Conv2d(64, 1, 1, 1, 0)
        #初始化
        nn.init.xavier_normal_(self.conv1x1.weight)
        nn.init.zeros_(self.conv1x1.bias)


    def forward(self, x):
        #前向传播
        #卷积
        x1 = self.conv1(x) #64*640*640
        #下采样，2×2最大池化
        x2 = F.max_pool2d(x1, 2) #64*320*320
        #卷积
        x2 = self.conv2(x2) #128*320*320
        #下采样，2×2最大池化
        x3 = F.max_pool2d(x2, 2) #128*160*160
        #卷积
        x3 = self.conv3(x3) #256*160*160
        #下采样，2×2最大池化
        x4 = F.max_pool2d(x3, 2) #256*80*80
        #卷积
        x4 = self.conv4(x4) #512*80*80
        #下采样，2×2最大池化
        x5 = F.max_pool2d(x4, 2) #512*40*40
        #卷积
        x5 = self.conv5(x5) #1024*40*40
        #上采样
        x6 = self.upsample1(x5) #512*80*80
        #拼接，卷积 1024*40*40

        x6 = torch.cat([x6,x4],dim=1)  #1024*80*80
        x6 = self.conv6(x6) #512*80*80
        #上采样
        x7 = self.upsample2(x6) #256*160*160
        x7 = torch.cat([x7, x3], dim=1) #512*160*160
        x7 = self.conv7(x7) #256*160*160
        #上采样
        x8 = self.upsample3(x7) #128*320*320
        x8 = torch.cat([x8, x2], dim=1) #256*320*320
        x8 = self.conv8(x8) #128*320*320
        #上采样
        x9 = self.upsample4(x8) #64*640*640
        x9 = torch.cat([x9, x1],dim=1) #128*640*640
        x9 = self.conv9(x9) #64*640*640
        #转为1通道
        result = self.conv1x1(x9)
        result = F.sigmoid(result)
        #转换为概率
        return result