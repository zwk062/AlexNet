import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3,224,224]   output[48,55,55]
            # 卷积核个数缩96减为48,下边每个卷积都会缩减为原论文的一半
            # padding如果传入tuple:(1,2) 1代表上下方各补一行零 2代表左右各补两列零
            # 如果需要像原论文实现，就需要调用nn.ZeroPad((1,2,1,2))左1右2上1下2
            # 为了方便直接写个2，N发现计算后不是整数，会成左2右1，会和原论文差不多
            nn.ReLU(inplace=True),  # 增加计算量，但降低内存使用,通过这方法可以将内存中载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 随机失活的比例
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )
        if init_weights:  # 只是讲初始化方法，其实pytorch会自动初始化
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 可见Sequential的好处了
        x = torch.flatten(x, start_dim=1)  # 第零维为batch不动，从第一维开始展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # Returns an iterator over all modules in the network
            if isinstance(m, nn.Conv2d):  # m是否是nn.Conv2d
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 如果偏置不为空
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化
                nn.init.constant_(m.bias, 0)
