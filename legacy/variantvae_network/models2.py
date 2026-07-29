import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, weight_decay=0.00005, strides=1, d=1, batch_momentum=0.99):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1, stride=strides, padding=0)
        self.bn1 = nn.BatchNorm3d(in_channels // 4, momentum=batch_momentum)
        
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=d, dilation=d)
        self.bn2 = nn.BatchNorm3d(in_channels // 4, momentum=batch_momentum)
        
        self.conv3 = nn.Conv3d(in_channels // 4, in_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm3d(in_channels, momentum=batch_momentum)
        
        self.shortcut = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=strides)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.1)
        out += residual
        return F.leaky_relu(out, negative_slope=0.1)

class Encoder(nn.Module):
    def __init__(self,cf):
        super(Encoder, self).__init__()
        self.cf = cf
        f1, f2, f3 = self.cf['enc_chs']
        layers = []
        layers.append(nn.Conv3d(self.cf['out_ch'], f1, kernel_size=3, stride=2, padding=1))
        layers.append(ResBlock(f1))
        layers.append(ResBlock(f1))
        layers.append(nn.Conv3d(f1, f2, kernel_size=3, stride=2, padding=1))
        layers.append(ResBlock(f2))
        layers.append(ResBlock(f2))
        layers.append(nn.Conv3d(f2, f3, kernel_size=3, stride=2, padding=0))
        layers.append(ResBlock(f3))
        layers.append(ResBlock(f3))
        layers.append(nn.Conv3d(f3, f3, kernel_size=3, stride=2, padding=1))
        layers.append(ResBlock(f3))
        layers.append(ResBlock(f3))
        layers.append(nn.Conv3d(f3, f3, kernel_size=3, stride=2, padding=1))
        layers.append(ResBlock(f3))
        self.encoding = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(f3 * 3 * 3 * 3, self.cf['z_dim'])
        self.fc_var = nn.Linear(f3 * 3 * 3 * 3, self.cf['z_dim'])

    def forward(self,x):
        fea = self.encoding(x)
        # print(f'shape of fea is {fea.shape}')
        fea = torch.flatten(fea, start_dim=1)   # (b, C, H, W) --> (batch, C*H*W)
        mu = self.fc_mu(fea)
        var = self.fc_var(fea)
        var = F.softplus(var)
        return mu, var

class Decoder(nn.Module):
    def __init__(self,cf):
        super(Decoder, self).__init__()
        self.cf = cf
        f1, f2, f3 = self.cf['enc_chs']
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        
        self.fc = nn.Linear(self.cf['z_dim'], 3 * 3 * 3 * f3)
        self.deconv_layers = nn.Sequential(
            ResBlock(f3),
            nn.ConvTranspose3d(f3, f3, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(f3), ResBlock(f3),
            nn.ConvTranspose3d(f3, f3, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(f3), ResBlock(f3),
            nn.ConvTranspose3d(f3, f2, kernel_size=3, stride=2, padding=0, output_padding=0),
            ResBlock(f2), ResBlock(f2),
            nn.ConvTranspose3d(f2, f1, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(f1), ResBlock(f1),
            nn.ConvTranspose3d(f1, self.cf['out_ch'], kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self,z):
        z = self.fc(z)
        z = z.reshape((-1,self.f3,3,3,3))
        z = self.deconv_layers(z)
        return z


class GMVAE2_3D(nn.Module):
    def __init__(self, cf):
        super(GMVAE2_3D, self).__init__()
        self.cf = cf
        f1, f2, f3 = self.cf['enc_chs']
        self.encoder = Encoder(cf)
        self.decoder = Decoder(cf)
        self.fc_mu = nn.Linear(f3 * 3 * 3 * 3, self.cf['z_dim'])
        self.fc_var = nn.Linear(f3 * 3 * 3 * 3, self.cf['z_dim'])
    
    def reparameterize(self, mean, var):
        print(f'mean shape is {mean.shape}\n')
        print(f'var shape is {var.shape}\n')
        eps = torch.randn_like(mean)
        std = var.sqrt()
        return eps * std + mean
    
    def sample(self, n, device):
        zm_prior = [torch.zeros((1, self.cf['z_dim']), requires_grad=False).to(device) for _ in range(1)]
        zv_prior = [torch.ones((1, self.cf['z_dim']), requires_grad=False).to(device) for _ in range(1)]
        eps = torch.randn((n, self.cf['z_dim'])).to(device)
        eps = eps * zv_prior[0].sqrt() + zm_prior[0]
        x_rec = torch.sigmoid(self.decoder(eps))
        x_rec = torch.round(x_rec*256)
        return x_rec, eps

    def forward(self, x):
        zm,zv = self.encoder(x)
        z = self.reparameterize(zm,zv)
        x_rec_logit = self.decoder(z)
        x_rec = torch.sigmoid(x_rec_logit)
        return x_rec, z, zm, zv

def log_normal(x, mu, var):
    return -0.5 * torch.sum(torch.log(2 * torch.pi) + torch.log(var) + (x - mu) ** 2 / var, dim=-1)