import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .blocks import ConvBlock, ResidualBlock, CBAM3D

def compute_feature_sizes(ori_size, num_stages, downsample_type, dim=3):
    """
    Compute feature map sizes for each stage of encoder/decoder.
    Args:
        ori_size (list): Original spatial size, e.g., [100, 100, 100].
        num_stages (int): Number of downsampling/upsampling stages.
        downsample_type (str): 'MaxPool', 'AvgPool', 'Stride2', or 'Stride4'.
        dim (int): 2 or 3 for 2D/3D.
    Returns:
        sizes (list): List of sizes per stage, starting from input size.
        output_paddings (list): Output padding for upsampling to match sizes.
    """
    sizes = [ori_size]
    current_size = ori_size.copy()
    
    for i in range(num_stages):
        if downsample_type in ['MaxPool', 'AvgPool']:
            current_size = [s // 2 for s in current_size]
        elif downsample_type == 'Stride2':
            current_size = [(s - 1) // 2 + 1 for s in current_size]
        elif downsample_type == 'Stride4':
            current_size = [(s - 1) // 4 + 1 for s in current_size]
        sizes.append(current_size.copy())
        print(f"Stage {i+1} downsample: {current_size}")
    
    expected_encoded_size = [2] * dim
    if sizes[-1] != expected_encoded_size:
        print(f"Warning: encoded size {sizes[-1]} != {expected_encoded_size}")
    
    output_paddings = []
    sizes_reversed = sizes[::-1]
    for i in range(num_stages):
        input_size = sizes_reversed[i]
        target_size = sizes[num_stages - i - 1]
        stride = 4 if downsample_type == 'Stride4' else 2
        kernel_size = 3
        padding = 1
        out_size = [
            (in_s - 1) * stride - 2 * padding + kernel_size
            for in_s in input_size
        ]
        output_padding = [
            target_s - out_s for target_s, out_s in zip(target_size, out_size)
        ]
        output_padding = [max(0, min(p, stride - 1)) for p in output_padding]
        output_paddings.append(output_padding)
        print(f"Upsample stage {i+1} padding: {output_padding}")
    
    return sizes, output_paddings



def get_trans_mode(cf):
    dim = cf['dim']
    assert dim in (2, 3), 'Only 2D or 3D supported'

    conv_block = ''
    if cf['normalization'] == 'Batch':
        conv_block += 'B'
    elif cf['normalization'] == 'Instance':
        conv_block += 'I'

    if cf['activation'] == 'Relu':
        conv_block += 'R'
    elif cf['activation'] == 'LeakyRelu':
        conv_block += 'L'

    conv_mode = 'C' + conv_block
    trans_mode = 'T' + conv_block
    return dim, conv_mode, trans_mode



class Encoder(nn.Module):
    def __init__(self, cf, embedding=False, spatial_latent=False):
        super().__init__()
        self.embedding = embedding
        self.cf = cf
        self.spatial_latent = spatial_latent
        block = ResidualBlock if cf['residual_block'] else ConvBlock
        dim, conv_mode, _ = get_trans_mode(cf)
        blocks_per_level = cf['blocks_per_level']
        enc_chs = cf['enc_chs']

        self.feature_sizes, _ = compute_feature_sizes(
            cf['ori_size'], len(enc_chs) + 1, cf['downsample'], dim
        )
        print(f"Encoder sizes: {self.feature_sizes}")

        self.encoding = nn.Sequential()
        stride = 2 if cf['downsample'] == 'Stride2' else 4 if cf['downsample'] == 'Stride4' else 1
        self.encoding.add_module(
            'InitialConv',
            ConvBlock(dim, cf['in_ch'], enc_chs[0], mode=conv_mode, stride=stride, kernel_size=3, padding=1)
        )
        pre_ch = enc_chs[0]

        for i, cur_ch in enumerate(enc_chs):
            for j in range(blocks_per_level):
                self.encoding.add_module(
                    f'Block{i}_{j}',
                    block(dim, pre_ch, cur_ch, mode=conv_mode, padding=1)
                )
                pre_ch = cur_ch

            if cf['downsample'] == 'MaxPool':
                MaxPool = getattr(nn, f'MaxPool{dim}d')
                self.encoding.add_module(f'Down{i}', MaxPool(2))
            elif cf['downsample'] == 'AvgPool':
                AvgPool = getattr(nn, f'AvgPool{dim}d')
                self.encoding.add_module(f'Down{i}', AvgPool(2))
            elif cf['downsample'] == 'Stride2':
                self.encoding.add_module(
                    f'Down{i}',
                    ConvBlock(dim, pre_ch, pre_ch, stride=2, mode=conv_mode, kernel_size=3, padding=1)
                )
            elif cf['downsample'] == 'Stride4':
                self.encoding.add_module(
                    f'Down{i}',
                    ConvBlock(dim, pre_ch, pre_ch, stride=4, mode=conv_mode, kernel_size=3, padding=1)
                )

        bottle_ch = cf['bottle_ch'] if cf['bottle_ch'] is not None else enc_chs[-1]
        self.encoding.add_module(
            'Bottleneck',
            block(dim, pre_ch, bottle_ch, mode=conv_mode, padding=1)
        )

        self.encoded_size = [bottle_ch] + self.feature_sizes[-1]
        print(f"Encoded size: {self.encoded_size}")
        
        if not spatial_latent:
            encoded_dim = np.prod(self.encoded_size)
            self.fc_mu = nn.Linear(encoded_dim, cf['z_dim'])
            self.fc_log_var = nn.Linear(encoded_dim, cf['z_dim'])

    def forward(self, x):
        embeddings = [] if self.embedding else None
        fea = x
        for name, layer in self.encoding.named_children():
            fea = layer(fea)
            if self.embedding and 'Down' in name:
                embeddings.append(fea)

        expected_size = tuple([x.shape[0]] + self.encoded_size)
        if fea.shape != expected_size:
            raise ValueError(f"Encoded shape {fea.shape} != {expected_size}")

        if self.spatial_latent:
            # For VQ-VAE, return spatial feature map
            mu = fea  # Shape: [batch_size, bottle_ch, 2, 2, 2]
            log_var = torch.zeros_like(mu)  # Dummy log_var for compatibility
        else:
            # For VAE/CVAE, flatten and use FC layers
            fea = fea.flatten(start_dim=1)
            mu = self.fc_mu(fea)
            log_var = self.fc_log_var(fea)

        return (mu, log_var, embeddings) if self.embedding else (mu, log_var)

class Decoder(nn.Module):
    def __init__(self, cf, input_ch=None):
        super().__init__()
        self.cf = cf
        bottle_ch = input_ch if input_ch is not None else (cf['bottle_ch'] if cf['bottle_ch'] is not None else cf['enc_chs'][-1])
        dim, conv_mode, trans_mode = get_trans_mode(cf)
        blocks_per_level = cf['blocks_per_level']
        dec_chs = cf['dec_chs']
        dec_chs = dec_chs + [dec_chs[-1]]
        print(f"Decoder channels: {dec_chs}")
        self.feature_sizes, self.output_paddings = compute_feature_sizes(
            cf['ori_size'], len(cf['enc_chs']) + 1, cf['downsample'], dim
        )
        print(f"Decoder sizes: {self.feature_sizes}")
        print(f"Output paddings: {self.output_paddings}")

        self.encoded_size = [bottle_ch] + self.feature_sizes[-1]
        encoded_dim = np.prod(self.encoded_size)
        self.fc_recover = nn.Linear(cf['z_dim'], encoded_dim)

        decoding = []
        pre_ch = bottle_ch

        for i, (cur_ch, output_padding) in enumerate(zip(dec_chs, self.output_paddings)):
            if cf['upsample'] in ['Stride2', 'Stride4']:
                stride = 4 if cf['upsample'] == 'Stride4' else 2
                decoding.append(
                    ConvBlock(
                        dim, pre_ch, cur_ch, stride=stride, output_padding=output_padding,
                        kernel_size=3, padding=1, mode=trans_mode
                    )
                )
                pre_ch = cur_ch

            block = ResidualBlock if cf['residual_block'] else ConvBlock
            for j in range(blocks_per_level):
                decoding.append(
                    block(dim, pre_ch, cur_ch, mode=conv_mode, padding=1)
                )

        decoding.append(ConvBlock(dim, pre_ch, cf['out_ch'], mode='C', padding=1))

        if cf['final_activation'] == 'Relu':
            decoding.append(nn.ReLU())
        elif cf['final_activation'] == 'Sigmoid':
            decoding.append(nn.Sigmoid())
        elif cf['final_activation'] == 'Softmax':
            decoding.append(nn.Softmax(dim=1))
        elif cf['final_activation'] == 'Identity':
            decoding.append(nn.Identity())

        self.decoding = nn.Sequential(*decoding)

    def forward(self, x):
        if x.dim() == 2:  # Flat latent vector [batch_size, z_dim]
            fea = self.fc_recover(x)
            fea = fea.view(-1, *self.encoded_size)
        else:  # Spatial feature map [batch_size, bottle_ch, 2, 2, 2]
            fea = x
        for layer in self.decoding:
            fea = layer(fea)
        x = fea

        if x.shape[2:] != tuple(self.cf['ori_size']):
            tz, ty, tx = self.cf['ori_size']
            x = x[:, :, :tz, :ty, :tx]

        expected_size = (self.cf['out_ch'], *self.cf['ori_size'])
        if x.shape[1:] != expected_size:
            raise ValueError(f"Output shape {x.shape[1:]} != {expected_size}")

        # if self.cf['final_activation'] == 'Softmax' and self.cf['train_label']:
        #     channel_sum = x.sum(dim=1)
        #     if not torch.allclose(channel_sum, torch.ones_like(channel_sum), rtol=1e-3):
        #         x = F.softmax(x, dim=1)
        return x

class VAE(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.z_dim = cf['z_dim']
        self.embedding = embedding
        self.encoder = Encoder(cf, embedding, spatial_latent=False)
        self.decoder = Decoder(cf)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z_vae = mu + torch.randn_like(std) * std
        return z_vae

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, deterministic=False):
        result = self.encode(x)
        mu, log_var = result[:2]
        embeddings = result[2] if self.embedding else None
        z = mu if deterministic else self.reparameterize(mu, log_var)
        out = self.decode(z)
        return (out, z, mu, log_var, embeddings) if self.embedding else (out, z, mu, log_var)

class CVAE(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.z_dim = cf['z_dim']
        self.embedding = embedding
        self.cond_dim = cf.get('cond_dim', cf['out_ch'])
        self.encoder = CVAEEncoder(cf, embedding)
        self.decoder = CVAEDecoder(cf)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z_vae = mu + torch.randn_like(std) * std
        return z_vae
    
    def sample(self, num_samples, device, cond):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decoder(z, cond), z
    
    def encode(self, x, cond):
        return self.encoder(x, cond)
    
    def decode(self, z, cond):
        return self.decoder(z, cond)
    
    def forward(self, x, cond, deterministic=False):
        result = self.encode(x, cond)
        mu, log_var = result[:2]
        embeddings = result[2] if self.embedding else None
        z = mu if deterministic else self.reparameterize(mu, log_var)
        out = self.decode(z, cond)
        return (out, z, mu, log_var, embeddings) if self.embedding else (out, z, mu, log_var)

class CVAEEncoder(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.embedding = embedding
        self.cf = cf
        block = ResidualBlock if cf['residual_block'] else ConvBlock
        dim, conv_mode, _ = get_trans_mode(cf)
        blocks_per_level = cf['blocks_per_level']
        enc_chs = cf['enc_chs']
        self.cond_dim = cf.get('cond_dim', cf['out_ch'])
        
        self.feature_sizes, _ = compute_feature_sizes(
            cf['ori_size'], len(enc_chs), cf['downsample'], dim
        )
        self.encoding = nn.Sequential()
        self.encoding.add_module(
            'InitialConv',
            ConvBlock(dim, cf['in_ch'] + self.cond_dim, enc_chs[0], mode=conv_mode, kernel_size=3, padding=1)
        )
        pre_ch = enc_chs[0]
        
        for i, cur_ch in enumerate(enc_chs):
            for j in range(blocks_per_level):
                self.encoding.add_module(
                    f'Block{i}_{j}',
                    block(dim, pre_ch, cur_ch, mode=conv_mode, padding=1)
                )
                pre_ch = cur_ch
            if cf['downsample'] == 'MaxPool':
                MaxPool = getattr(nn, f'MaxPool{dim}d')
                self.encoding.add_module(f'Down{i}', MaxPool(2))
            elif cf['downsample'] == 'AvgPool':
                AvgPool = getattr(nn, f'AvgPool{dim}d')
                self.encoding.add_module(f'Down{i}', AvgPool(2))
            elif cf['downsample'] == 'Stride2':
                self.encoding.add_module(
                    f'Down{i}',
                    ConvBlock(dim, pre_ch, pre_ch, stride=2, mode=conv_mode, kernel_size=3, padding=1)
                )
            elif cf['downsample'] == 'Stride4':
                self.encoding.add_module(
                    f'Down{i}',
                    ConvBlock(dim, pre_ch, pre_ch, stride=4, mode=conv_mode, kernel_size=3, padding=1)
                )
        
        bottle_ch = cf['bottle_ch'] if cf['bottle_ch'] is not None else enc_chs[-1]
        self.encoding.add_module(
            'Bottleneck',
            block(dim, pre_ch, bottle_ch, mode=conv_mode, padding=1)
        )
        
        self.encoded_size = [bottle_ch] + self.feature_sizes[-1]
        encoded_dim = np.prod(self.encoded_size)
        self.fc_mu = nn.Linear(encoded_dim, cf['z_dim'])
        self.fc_log_var = nn.Linear(encoded_dim, cf['z_dim'])
    
    def forward(self, x, cond):
        embeddings = [] if self.embedding else None
        cond = cond.view(-1, self.cond_dim, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, cond], dim=1)
        fea = x
        for name, layer in self.encoding.named_children():
            fea = layer(fea)
            if self.embedding and 'Down' in name:
                embeddings.append(fea)
        
        expected_size = tuple([x.shape[0]] + self.encoded_size)
        if fea.shape != expected_size:
            raise ValueError(f"Encoded shape {fea.shape} != {expected_size}")
        
        fea = fea.flatten(start_dim=1)
        mu = self.fc_mu(fea)
        log_var = self.fc_log_var(fea)
        return (mu, log_var, embeddings) if self.embedding else (mu, log_var)

class CVAEDecoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf
        bottle_ch = cf['bottle_ch'] if cf['bottle_ch'] is not None else cf['enc_chs'][-1]
        dim, conv_mode, trans_mode = get_trans_mode(cf)
        blocks_per_level = cf['blocks_per_level']
        dec_chs = cf['dec_chs']
        self.cond_dim = cf.get('cond_dim', cf['out_ch'])
        
        self.feature_sizes, self.output_paddings = compute_feature_sizes(
            cf['ori_size'], len(cf['enc_chs']), cf['downsample'], dim
        )
        self.encoded_size = [bottle_ch] + self.feature_sizes[-1]
        encoded_dim = np.prod(self.encoded_size)
        self.fc_recover = nn.Linear(cf['z_dim'] + self.cond_dim, encoded_dim)
        
        decoding = []
        pre_ch = bottle_ch
        for i, (cur_ch, output_padding) in enumerate(zip(dec_chs, self.output_paddings)):
            if cf['upsample'] in ['Stride2', 'Stride4']:
                stride = 4 if cf['upsample'] == 'Stride4' else 2
                decoding.append(
                    ConvBlock(
                        dim, pre_ch, cur_ch, stride=stride, output_padding=output_padding,
                        kernel_size=3, padding=1, mode=trans_mode
                    )
                )
                pre_ch = cur_ch
            
            block = ResidualBlock if cf['residual_block'] else ConvBlock
            for j in range(blocks_per_level):
                decoding.append(
                    block(dim, pre_ch, cur_ch, mode=conv_mode, padding=1)
                )
        
        decoding.append(ConvBlock(dim, pre_ch, cf['out_ch'], mode='C', padding=1))
        
        if cf['final_activation'] == 'Relu':
            decoding.append(nn.ReLU())
        elif cf['final_activation'] == 'Sigmoid':
            decoding.append(nn.Sigmoid())
        elif cf['final_activation'] == 'Softmax':
            decoding.append(nn.Softmax(dim=1))
        elif cf['final_activation'] == 'Identity':
            decoding.append(nn.Identity())
        
        self.decoding = nn.Sequential(*decoding)
    
    def forward(self, z, cond):
        print(f'cvae z shape is {z.shape}, cond shape is {cond.shape}')
        z = torch.cat([z, cond], dim=1)
        fea = self.fc_recover(z)
        fea = fea.view(-1, *self.encoded_size)
        for layer in self.decoding:
            fea = layer(fea)
        x = fea
        
        if x.shape[2:] != tuple(self.cf['ori_size']):
            tz, ty, tx = self.cf['ori_size']
            x = x[:, :, :tz, :ty, :tx]
        
        expected_size = (self.cf['out_ch'], *self.cf['ori_size'])
        if x.shape[1:] != expected_size:
            raise ValueError(f"Output shape {x.shape[1:]} != {expected_size}")
        
        # if self.cf['final_activation'] == 'Softmax':
        #     channel_sum = x.sum(dim=1)
        #     if not torch.allclose(channel_sum, torch.ones_like(channel_sum), rtol=1e-3):
        #         x = F.softmax(x, dim=1)
        return x

class VQVAE(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.z_dim = cf['z_dim']
        self.num_embeddings = cf.get('num_embeddings', 512)
        self.embedding_dim = cf['bottle_ch'] if cf['bottle_ch'] is not None else cf['enc_chs'][-1]
        self.commitment_cost = cf.get('commitment_cost', 0.25)
        self.embedding = embedding
        self.encoder = Encoder(cf, embedding, spatial_latent=cf.get('vqvae_spatial_latent', True))
        self.decoder = Decoder(cf, input_ch=self.embedding_dim)
        
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
    
    def quantize(self, z):
        # z: [batch_size, bottle_ch, 2, 2, 2]
        batch_size, channels, d, h, w = z.shape
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, channels)  # [batch_size*d*h*w, bottle_ch]
        # print(f'z shape is {z.shape}')
        # print(f'z_flattened shape is {z_flattened.shape}')
        # print(f'codebook weight t shape is {self.codebook.weight.t().shape}')
        # print(f'codebook weight is {self.codebook.weight}')
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                     torch.sum(self.codebook.weight ** 2, dim=1) -
                     2 * torch.matmul(z_flattened, self.codebook.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.codebook.weight).view(batch_size, d, h, w, self.embedding_dim)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # [batch_size, embedding_dim, d, h, w]
        
        commitment_loss = self.commitment_cost * torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        return quantized, commitment_loss, codebook_loss, encodings
    
    def forward(self, x, deterministic=False):
        mu, log_var = self.encoder(x)
        # Standard VQ-VAE quantizes the encoder feature map directly.
        # Keep stochastic sampling optional for backwards compatibility,
        # but default it OFF because spatial_latent returns dummy log_var=0.
        if deterministic or not self.cf.get('vqvae_stochastic', False):
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        quantized, commitment_loss, codebook_loss, encodings = self.quantize(z)
        if self.cf.get('debug_shapes', False):
            print(f'quantized shape is {quantized.shape}')
        x_hat = self.decoder(quantized)
        return x_hat, quantized, mu, log_var, commitment_loss, codebook_loss
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    
    def sample(self, num_samples, device):
        indices = torch.randint(0, self.num_embeddings, (num_samples, *self.encoder.feature_sizes[-1]), device=device)
        encodings = torch.zeros(indices.shape[0], *indices.shape[1:], self.num_embeddings, device=device)
        encodings.scatter_(-1, indices.unsqueeze(-1), 1)
        quantized = torch.matmul(encodings.view(-1, self.num_embeddings), self.codebook.weight)
        quantized = quantized.view(num_samples, *self.encoder.feature_sizes[-1], self.embedding_dim)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # [num_samples, z_dim, 2, 2, 2]
        return self.decoder(quantized), quantized
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

class HybridVAEUNet(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.z_dim = cf['z_dim']
        self.embedding = embedding
        self.encoder = Encoder(cf, embedding, spatial_latent=False)
        self.vae_decoder = Decoder(cf)
        self.unet_decoder = UNetDecoder(cf)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim).to(device)
        vae_out = self.vae_decoder(z)
        unet_out = self.unet_decoder(z)
        return vae_out, unet_out, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        vae_out = self.vae_decoder(z)
        unet_out = self.unet_decoder(z)
        return vae_out, unet_out
    
    def forward(self, x, deterministic=False):
        result = self.encode(x)
        mu, log_var = result[:2]
        embeddings = result[2] if self.embedding else None
        z = mu if deterministic else self.reparameterize(mu, log_var)
        print(f'z shape is {z.shape}')
        vae_out, unet_out = self.decode(z)
        return (vae_out, unet_out, z, mu, log_var, embeddings) if self.embedding else (vae_out, unet_out, z, mu, log_var)

class UNetDecoder(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf
        bottle_ch = cf['bottle_ch'] if cf['bottle_ch'] is not None else cf['enc_chs'][-1]
        dim, conv_mode, trans_mode = get_trans_mode(cf)
        blocks_per_level = cf['blocks_per_level']
        dec_chs = cf['dec_chs']
        
        self.feature_sizes, self.output_paddings = compute_feature_sizes(
            cf['ori_size'], len(cf['enc_chs']), cf['downsample'], dim
        )
        self.encoded_size = [bottle_ch] + self.feature_sizes[-1]
        encoded_dim = np.prod(self.encoded_size)
        self.fc_recover = nn.Linear(cf['z_dim'], encoded_dim)
        
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        pre_ch = bottle_ch  # Start with bottle_ch (e.g., 128)
        for i, (cur_ch, output_padding) in enumerate(zip(dec_chs, self.output_paddings)):
            # Use pre_ch as input channels (no skip connections)
            in_channels = pre_ch
            self.upconvs.append(
                ConvBlock(
                    dim, in_channels, cur_ch, stride=2, 
                    output_padding=output_padding, kernel_size=3, padding=1, mode=trans_mode
                )
            )
            block = ResidualBlock if cf['residual_block'] else ConvBlock
            block_list = nn.ModuleList()
            for j in range(blocks_per_level):
                block_list.append(block(dim, cur_ch, cur_ch, mode=conv_mode, padding=1))
            self.blocks.append(block_list)
            pre_ch = cur_ch  # Update pre_ch to current layer's output channels
        
        print(f'upconvs are {self.upconvs}')
        self.final_conv = ConvBlock(dim, pre_ch, cf['out_ch'], mode='C', padding=1)
        if cf['final_activation'] == 'Softmax':
            self.final_act = nn.Softmax(dim=1)
        elif cf['final_activation'] == 'Sigmoid':
            self.final_act = nn.Sigmoid()
        elif cf['final_activation'] == 'Identity':
            self.final_act = nn.Identity()
        else:
            self.final_act = nn.ReLU()
    
    def forward(self, z, skip_connections=None):
        fea = self.fc_recover(z.view(z.shape[0], -1))
        fea = fea.view(-1, *self.encoded_size)
        for i, (upconv, block_list) in enumerate(zip(self.upconvs, self.blocks)):
            fea = upconv(fea)  # No skip connections, so no concatenation
            for block in block_list:
                fea = block(fea)
        x = self.final_conv(fea)
        x = self.final_act(x)
        
        if x.shape[2:] != tuple(self.cf['ori_size']):
            tz, ty, tx = self.cf['ori_size']
            x = x[:, :, :tz, :ty, :tx]
        
        expected_size = (self.cf['out_ch'], *self.cf['ori_size'])
        if x.shape[1:] != expected_size:
            raise ValueError(f"Output shape {x.shape[1:]} != {expected_size}")
        
        return x

class GMVAE2_3D(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()
        self.cf = cf
        self.K = cf['K']
        self.z_dim = cf['z_dim']
        self.embedding = embedding
        # Learnable prior parameters (mixture components)
        limit = (3 / self.z_dim) ** 0.5
        self.zm_prior = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.z_dim).uniform_(-limit, limit)) for _ in range(self.K)
        ])
        self.zv_prior = nn.ParameterList([
            nn.Parameter(torch.ones(1, self.z_dim), requires_grad=False) for _ in range(self.K)
        ])

        # Shared encoder and decoder
        self.encoder = Encoder(cf, embedding=embedding, spatial_latent=False)
        self.decoder = Decoder(cf)

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return mu, torch.exp(log_var)

    def reparameterize(self, mu, var):
        eps = torch.randn_like(mu)
        return mu + eps * var.sqrt()

    def decode(self, z):
        return self.decoder(z)

    def log_normal(self, x, mu, var):
        return -0.5 * torch.sum(
            torch.log(2 * torch.pi * var) + (x - mu) ** 2 / var, dim=-1
        )

    def q_y(self, z):
        logits = []
        for i in range(self.K):
            mu_k = self.zm_prior[i]
            var_k = self.zv_prior[i]
            log_p_k = self.log_normal(z, mu_k, var_k) + np.log(1.0 / self.K)
            logits.append(log_p_k)
        logits = torch.stack(logits, dim=-1)  # [B, K]
        qy = torch.softmax(logits, dim=-1)
        return logits, qy, torch.argmax(qy, dim=-1)

    def sample(self, n, device):
        reconstructions = []
        for i in range(self.K):
            eps = torch.randn(n, self.z_dim).to(device)
            z = self.zm_prior[i].to(device) + eps * self.zv_prior[i].to(device).sqrt()
            x_hat = torch.sigmoid(self.decode(z))
            reconstructions.append(x_hat)
        return reconstructions

    def forward(self, x):
        mu, var = self.encode(x)           # [B, z_dim]
        z = self.reparameterize(mu, var)   # [B, z_dim]
        logits_y, qy, y_idx = self.q_y(z)  # [B, K]
        x_hat = self.decode(z)
        return x_hat, z, mu, var, logits_y, qy, y_idx

class EncoderCBAM(Encoder):
    def __init__(self, cf, embedding=False, spatial_latent=False):
        super().__init__(cf, embedding, spatial_latent)

        if not cf.get('use_cbam', True):
            return

        reduction = cf.get('cbam_reduction', 16)

        # Wrap all ConvBlock / ResidualBlock modules with CBAM
        for name, module in self.encoding.named_children():

            if isinstance(module, (ConvBlock, ResidualBlock)):
                out_ch = module.out_channels if hasattr(module, "out_channels") else None

                if out_ch is None:
                    continue

                wrapped = nn.Sequential(
                    module,
                    CBAM3D(out_ch, reduction=reduction)
                )

                setattr(self.encoding, name, wrapped)

class DecoderCBAM(Decoder):
    def __init__(self, cf, input_ch=None):
        super().__init__(cf, input_ch)

        if not cf.get('use_cbam', True):
            return

        reduction = cf.get('cbam_reduction', 16)

        new_layers = []

        for layer in self.decoding:

            if isinstance(layer, (ConvBlock, ResidualBlock)):
                out_ch = layer.out_channels if hasattr(layer, "out_channels") else None

                if out_ch is not None:
                    layer = nn.Sequential(
                        layer,
                        CBAM3D(out_ch, reduction=reduction)
                    )

            new_layers.append(layer)

        self.decoding = nn.Sequential(*new_layers)

class VAE_CBAM(nn.Module):
    def __init__(self, cf, embedding=False):
        super().__init__()

        self.z_dim = cf['z_dim']
        self.embedding = embedding

        self.encoder = EncoderCBAM(
            cf,
            embedding=embedding,
            spatial_latent=False
        )

        self.decoder = DecoderCBAM(cf)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim).to(device)
        recon = self.decoder(z)
        return recon, z
        
    def forward(self, x, deterministic=False):
        result = self.encode(x)
        mu, log_var = result[:2]
        embeddings = result[2] if self.embedding else None

        z = mu if deterministic else self.reparameterize(mu, log_var)

        out = self.decode(z)

        if self.embedding:
            return out, z, mu, log_var, embeddings
        else:
            return out, z, mu, log_var
