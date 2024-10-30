import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_size, act_fn='relu'):
        super(Encoder, self).__init__()
        self.act = getattr(F, act_fn)
        self.embedding_size = embedding_size

       
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 输入: (batch, 3, 64, 64) -> 输出: (batch, 64, 64, 64)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # (64, 64, 64) -> (128, 32, 32)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # (128, 32, 32) -> (256, 16, 16)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # (256, 16, 16) -> (512, 8, 8)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (512, 8, 8) -> (512, 1, 1)

        self.fc = nn.Linear(512, embedding_size)

    def forward(self, observation):
        out = self.conv1(observation)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act(out)

        out = self.global_avg_pool(out)  # (batch, 512, 1, 1)
        out = out.view(out.size(0), -1)  # (batch, 512)

        if self.embedding_size != 512:
            out = self.fc(out)  # (batch, embedding_size)

        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)  # Downsample by 2
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Downsample by 2
        x = self.layer1(x)  # No downsampling
        x = self.layer2(x)  # Downsample by 2
        x = self.layer3(x)  # Downsample by 2
        x = self.layer4(x)  # Downsample by 2
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pool to (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        x = self.linear(x)  # Map to latent space (batch_size, z_dim)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

# if __name__ == "__main__":
#     # Initialize encoder and decoder
#     encoder = ResNet18Enc(z_dim=256)
#     img = torch.randn(10, 3, 84, 84)
#     z = encoder(img)
#     print(z.shape)
        