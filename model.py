from torch import nn
import torch
import math

def load_model_weights(model, state_dict):
    model_state = model.state_dict()
    matched_weights = {
        k: v for k, v in state_dict.items() 
        if k in model_state and v.shape == model_state[k].shape
    }
    unmatched = set(model_state.keys()) - set(matched_weights.keys())
    if unmatched:
        print(f"Warning - Unmatched keys: {unmatched}")
    
    model.load_state_dict(matched_weights, strict=False)
    
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x_encoded = self.encoder(x).pooler_output
                
        x_predicted = self.decoder(x_encoded.view(self.batch_size, 1, self.up_state, self.up_state))
                        
        return x_predicted, y
    
def initialize_unet(model):
    def _init_weights(m):
                
        if isinstance(m, MultiHeadAttention):
            # Initialize the output projection for all attention types
            nn.init.xavier_uniform_(m.proj.weight, gain=0.01)
            
            if m.is_cross_attention:
                # Cross attention has separate Q and KV projections
                nn.init.xavier_uniform_(m.q_proj.weight, gain=0.01)
                nn.init.xavier_uniform_(m.kv_proj.weight, gain=0.01)
            else:
                # Self attention has combined QKV projection
                nn.init.xavier_uniform_(m.qkv.weight, gain=0.01)
                
            # Zero init all biases if they exist
            if m.proj.bias is not None:
                nn.init.zeros_(m.proj.bias)
                
            if m.is_cross_attention:
                if m.q_proj.bias is not None:
                    nn.init.zeros_(m.q_proj.bias)
                if m.kv_proj.bias is not None:
                    nn.init.zeros_(m.kv_proj.bias)
            else:
                if m.qkv.bias is not None:
                    nn.init.zeros_(m.qkv.bias)
    
    model.apply(_init_weights)
    return model
    
class OperatorLoss(nn.Module):
    def __init__(self, a1):
        super().__init__()
        
        self.mse = lambda x, y: torch.mean(torch.mean(torch.linalg.norm(x - y, dim=(2, 3)), dim=1))
        self.mae = lambda x, y: torch.mean(torch.linalg.norm(x - y, dim=1, ord=1))
        self.a1 = a1
        
    def forward(self, x_predicted, y):
        # Prediction/L2
        pred = self.mse(x_predicted, y)                       
        return self.a1 * pred 

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, height, width):
        mask = torch.zeros(1, height, width, self.dim)
        
        # Create position indices
        y_pos = torch.arange(0, height)[:, None, None].float() / height  # [H, 1, 1]
        x_pos = torch.arange(0, width)[None, :, None].float() / width    # [1, W, 1]
        
        # Calculate frequency bands
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        
        # Alternate between sin/cos using y_pos for even indices and x_pos for odd indices
        mask[..., 0::2] = torch.sin(y_pos * div_term)
        mask[..., 1::2] = torch.cos(x_pos * div_term)
        
        return mask.permute(0, 3, 1, 2)  # [1, C, H, W] for standard image format

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, latent_dim=None, is_cross_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.is_cross_attention = is_cross_attention
        self.attr_flag = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # Default latent dim if not specified
        if latent_dim is None:
            latent_dim = dim // 2
        self.latent_dim = latent_dim
        
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections based on attention type
        if is_cross_attention:
            self.q_proj = nn.Linear(dim, latent_dim)
            self.kv_proj = nn.Linear(dim, latent_dim * 2)
        else:
            # Combined QKV for self-attention
            self.qkv = nn.Linear(dim, latent_dim * 3)
            
        self.proj = nn.Linear(latent_dim, dim)
        self.pos_encoding = PositionalEncoding2D(dim)
                
    def forward(self, x, context=None):
        batch_size, channels, height, width = x.shape
        
        pos_enc = self.pos_encoding(height, width)
        pos_enc = pos_enc.expand(batch_size, -1, -1, -1).to(x.device)
        
        # Reshape to (B, H*W, C)
        x_flat = (x + pos_enc).view(batch_size, channels, -1).transpose(1, 2)
        
        if self.is_cross_attention:
            # Reshape context
            context = (context + pos_enc).view(batch_size, channels, -1).transpose(1, 2)
            
            # Separate Q, K, V projections for cross-attention
            q = self.q_proj(x_flat)
            kv = self.kv_proj(context)
            k, v = kv.chunk(2, dim=-1)
            
            # Reshape to multi-head format
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:     
            # Combined QKV projection for self-attention
            qkv = self.qkv(x_flat).chunk(3, dim=-1)
            q, k, v = map(lambda t: t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Use flash attention if available
        if self.attr_flag:
            out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        
        # Reshape and project back
        out = out.transpose(1, 2).reshape(batch_size, -1, self.latent_dim)
        out = self.proj(out)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).view(batch_size, channels, height, width)
        
        return out + x
    
class ConvNextUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dims=[96, 192, 384, 768]):
        super().__init__()
        
        # Define stage depths
        stage_depths = [3, 3, 9, 3]
        
        # Initial embedding with smaller stride
        self.embeddings = ConvNextEmbeddings(
            in_channels=in_channels,
            out_channels=dims[0],
            patch_size=2,
            stride=2
        )
        
        # Extra initial stage without downsampling
        self.initial_stage = ConvNextStage(
            in_channels=dims[0],
            out_channels=dims[0],
            depth=3,
            is_downsample=False
        )
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        for i in range(len(dims)):
            self.encoder_stages.append(
                ConvNextStage(
                    in_channels=dims[i-1] if i > 0 else dims[0],
                    out_channels=dims[i],
                    depth=stage_depths[i],
                    is_downsample=i > 0
                )
            )
        
        # Bottleneck self-attention
        self.bottleneck_attn = MultiHeadAttention(dims[-1])
        
        # Decoder stages with attention
        self.decoder_stages = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        reversed_dims = list(reversed(dims))
        reversed_depths = list(reversed(stage_depths[:-1]))
        
        for i in range(len(dims) - 1):
            self.decoder_stages.append(
                ConvNextStage(
                    in_channels=reversed_dims[i],
                    out_channels=reversed_dims[i + 1],
                    depth=reversed_depths[i],
                    is_upsample=True
                )
            )
            self.attentions.append(
                MultiHeadAttention(dim=reversed_dims[i + 1], is_cross_attention=True)
            )
        
        # Final upsampling and output
        self.final_upsample = nn.Sequential(
            ConvNextLayerNorm(),
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, stride=2),
            ConvNextLayer(dims[0], expand_ratio=4)
        )
        
        self.final_layer = nn.Sequential(
            ConvNextLayerNorm(),
            nn.Conv2d(dims[0], out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # Initial embedding (downsamples by 2)
        x = self.embeddings(x)
        x = self.initial_stage(x)
        
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)
            
        # Apply self-attention at bottleneck
        x = self.bottleneck_attn(x)
            
        # Remove last feature map as it's the bottleneck
        encoder_features = encoder_features[:-1]
        
        # Decoder path with skip connections and attention
        for i, (stage, attention) in enumerate(zip(self.decoder_stages, self.attentions)):
            # Upsample
            x = stage(x)
            
            # Apply attention between skip connection and upsampled features
            skip = encoder_features[-(i+1)]
            x = attention(skip, x)
        
        # Final upsampling to match input resolution
        x = self.final_upsample(x)
        
        # Normalize and trasnform to desired output channels
        return self.final_layer(x)

class ConvNextStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, is_downsample=False, is_upsample=False):
        super().__init__()
        
        if is_downsample:
            self.sampling = nn.Sequential(
                ConvNextLayerNorm(),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        elif is_upsample:
            self.sampling = nn.Sequential(
                ConvNextLayerNorm(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        else:
            # No sampling -> identity -> easy forward pass
            self.sampling = nn.Identity()
            if in_channels != out_channels:
                self.sampling = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
        self.layers = nn.Sequential(*[
            ConvNextLayer(
                dim=out_channels,
                expand_ratio=4
            ) for _ in range(depth)
        ])
        
    def forward(self, x):
        x = self.sampling(x)
        x = self.layers(x)
        return x

class GlobalResponseNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x shape: (B, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # Global feature response
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)   # Normalize
        return x * nx * (1 + self.gamma) + self.beta

class ConvNextLayer(nn.Module):
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextLayerNorm()
        self.pwconv1 = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(dim * expand_ratio)  # Note: dim * expand_ratio because it's after expansion
        self.pwconv2 = nn.Linear(dim * expand_ratio, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)            # GRN after GeLU
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = residual + self.drop_path(x)
        return x

class ConvNextEmbeddings(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, patch_size=2, stride=2):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=patch_size, 
            stride=stride
        )
        self.layernorm = ConvNextLayerNorm()
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNextLayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)