import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNet, self).__init__()
        self.fc = nn.Linear(1024, 1000)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            DSConv(32, 64, 3, 1, 1),
            DSConv(64, 128, 3, 2, 1),
            DSConv(128, 128, 3, 1, 1),
            DSConv(128, 256, 3, 2, 1),
            DSConv(256, 256, 3, 1, 1),
            DSConv(256, 512, 3, 2, 1),
            DSConv(512, 512, 3, 1, 1),
            DSConv(512, 512, 3, 1, 1),
            DSConv(512, 512, 3, 1, 1),
            DSConv(512, 512, 3, 1, 1),
            DSConv(512, 512, 3, 1, 1),
            DSConv(512, 1024, 3, 2, 1),
            DSConv(1024, 1024, 3, 2, 4),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=False):
        super(DSConv, self).__init__()
        self.batch_norm = batch_norm
        self.dconv = DConv(in_channels, kernel_size, stride, padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU(True)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dconv(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        return x


class DConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

    def forward(self, x):
        return self.dconv(x)


from torchsummary import summary
model = MobileNet(3, 1000).cuda()
summary(model, (3, 224, 224))
