import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    '''
    Decoder network for Dreamer
    From belief and state to observation
    '''
    def __init__(self, belief_size, state_size, embedding_size, act_fn='relu'):
        super(Decoder, self).__init__()
        self.act = getattr(F, act_fn)
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(belief_size + state_size, embedding_size * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(
            embedding_size, 128, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)

        self.deconv5 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, belief, state):
        combined = torch.cat([belief, state], dim=1)
        hidden = self.fc1(combined)
        hidden = hidden.view(-1, self.embedding_size, 4, 4)

        hidden = self.deconv1(hidden)
        hidden = self.bn1(hidden)
        hidden = self.act(hidden)

        hidden = self.deconv2(hidden)
        hidden = self.bn2(hidden)
        hidden = self.act(hidden)

        hidden = self.deconv3(hidden)
        hidden = self.bn3(hidden)
        hidden = self.act(hidden)

        hidden = self.deconv4(hidden)
        hidden = self.bn4(hidden)
        hidden = self.act(hidden)

        observation = self.deconv5(hidden)
        # observation = torch.sigmoid(observation)  # Apply sigmoid activation function here

        return observation

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlockDec, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        if self.stride == 2:
            # Use transposed convolution for upsampling
            self.conv1 = nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
            )
        
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Shortcut connection
        if self.stride != 1 or in_planes != out_planes:
            if self.stride == 2:
                # When upsampling, use ConvTranspose2d
                shortcut_conv = nn.ConvTranspose2d(
                    in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
                )
            else:
                # When not upsampling but channels differ, use Conv2d
                shortcut_conv = nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=1, bias=False
                )
            self.shortcut = nn.Sequential(
                shortcut_conv,
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        # Ensure that the spatial dimensions match
        if shortcut.size() != out.size():
            raise ValueError(f"Shortcut size {shortcut.size()} does not match output size {out.size()}")
        out += shortcut
        out = self.relu(out)
        return out

class ResNet18Dec(nn.Module):
    '''
    ResNet Decoder network for Dreamer
    From belief and state to observation
    '''
    def __init__(self, num_blocks=[2, 2, 2, 2], obs_dim=20, latent_act_dim=10, nc=3):
        super(ResNet18Dec, self).__init__()
        self.in_planes = 512
        z_dim = obs_dim + latent_act_dim
        self.linear = nn.Linear(z_dim, 512 * 7 * 7)  # Adjust the output size of the linear layer

        # Define decoder layers
        self.layer4 = self._make_layer(256, num_blocks[0], stride=2)  # 7x7 -> 14x14
        self.layer3 = self._make_layer(128, num_blocks[1], stride=2)  # 14x14 -> 28x28
        self.layer2 = self._make_layer(64, num_blocks[2], stride=1)   # 28x28 -> 28x28
        self.layer1 = self._make_layer(64, num_blocks[3], stride=1)   # 28x28 -> 28x28

        # Final transposed convolution layer, enlarging the size from (28, 28) to (84, 84)
        self.conv1 = nn.ConvTranspose2d(
            64, nc, kernel_size=4, stride=3, padding=1, output_padding=1, bias=False
        )

    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BasicBlockDec(self.in_planes, out_planes, stride=stride))
            else:
                layers.append(BasicBlockDec(out_planes, out_planes, stride=1))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, obs, latent_act):
        z = torch.cat([obs, latent_act], dim=1)
        x = self.linear(z)
        x = x.view(z.size(0), 512, 7, 7)  # Initial feature map size is (7, 7)

        x = self.layer4(x)   # Output size is (256, 14, 14)
        x = self.layer3(x)   # Output size is (128, 28, 28)
        x = self.layer2(x)   # Output size is (64, 28, 28)
        x = self.layer1(x)   # Output size is (64, 28, 28)

        x = self.conv1(x)    # Output size is (nc, 84, 84)
        x = torch.sigmoid(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResNet18DecV2(nn.Module):
    '''
    ResNet Decoder network for VAE/AE
    From latent space to observation
    '''
    def __init__(self, num_blocks=[2, 2, 2, 2], 
                 z_dim=10, nc=3):
        super(ResNet18DecV2, self).__init__()
        self.in_planes = 512
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 512 * 7 * 7)  # Adjust the output size of the linear layer

        # Define decoder layers
        self.layer4 = self._make_layer(256, num_blocks[0], stride=2)  # 7x7 -> 14x14
        self.layer3 = self._make_layer(128, num_blocks[1], stride=2)  # 14x14 -> 28x28
        self.layer2 = self._make_layer(64, num_blocks[2], stride=1)   # 28x28 -> 28x28
        self.layer1 = self._make_layer(64, num_blocks[3], stride=1)   # 28x28 -> 28x28

        # Final transposed convolution layer, enlarging the size from (28, 28) to (84, 84)
        self.conv1 = nn.ConvTranspose2d(
            64, nc, kernel_size=4, stride=3, padding=1, output_padding=1, bias=False
        )

    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BasicBlockDec(self.in_planes, out_planes, stride=stride))
            else:
                layers.append(BasicBlockDec(out_planes, out_planes, stride=1))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 7, 7)  # Initial feature map size is (7, 7)

        x = self.layer4(x)   # Output size is (256, 14, 14)
        x = self.layer3(x)   # Output size is (128, 28, 28)
        x = self.layer2(x)   # Output size is (64, 28, 28)
        x = self.layer1(x)   # Output size is (64, 28, 28)

        x = self.conv1(x)    # Output size is (nc, 84, 84)
        x = torch.sigmoid(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResNet18DecV3(nn.Module):
    '''
    ResNet Decoder network for VAE/AE
    From action and latent observation to frame difference
    output: frame difference [-1, 1]
    '''
    def __init__(self, num_blocks=[2, 2, 2, 2], 
                 action_dim=10, latent_obs_dim = 10, nc=3):
        super(ResNet18DecV3, self).__init__()
        self.in_planes = 512
        self.z_dim = action_dim + latent_obs_dim
        self.linear = nn.Linear(self.z_dim, 512 * 7 * 7)  # Adjust the output size of the linear layer

        # Define decoder layers
        self.layer4 = self._make_layer(256, num_blocks[0], stride=2)  # 7x7 -> 14x14
        self.layer3 = self._make_layer(128, num_blocks[1], stride=2)  # 14x14 -> 28x28
        self.layer2 = self._make_layer(64, num_blocks[2], stride=1)   # 28x28 -> 28x28
        self.layer1 = self._make_layer(64, num_blocks[3], stride=1)   # 28x28 -> 28x28

        # Final transposed convolution layer, enlarging the size from (28, 28) to (84, 84)
        self.conv1 = nn.ConvTranspose2d(
            64, nc, kernel_size=4, stride=3, padding=1, output_padding=1, bias=False
        )

    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BasicBlockDec(self.in_planes, out_planes, stride=stride))
            else:
                layers.append(BasicBlockDec(out_planes, out_planes, stride=1))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, act, latent_obs):
        z = torch.cat([act, latent_obs], dim=-1)
        x = self.linear(z)
        x = x.view(z.size(0), 512, 7, 7)  # Initial feature map size is (7, 7)

        x = self.layer4(x)   # Output size is (256, 14, 14)
        x = self.layer3(x)   # Output size is (128, 28, 28)
        x = self.layer2(x)   # Output size is (64, 28, 28)
        x = self.layer1(x)   # Output size is (64, 28, 28)

        x = self.conv1(x)    # Output size is (nc, 84, 84)
        x = torch.tanh(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    obs_dim = 60
    latent_act_dim = 40
    decoder = ResNet18Dec(obs_dim=obs_dim, latent_act_dim=latent_act_dim, nc=3)
    # Test output size
    sim_data = torch.randn(1, obs_dim)
    action = torch.randn(1, latent_act_dim)
    output = decoder(sim_data, action)
    print(output.shape)  # Expected output: torch.Size([1, 3, 84, 84])
