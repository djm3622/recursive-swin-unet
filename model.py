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

# uses the final step only
def standard_step(batch, model, criterion):
    x, y = batch
    out = model(x)
    loss = criterion(out, y)
        
    return loss

# TODO : step function to use all the intermeidate output 
# the loss is against the next step 
# (maybe cross attention could help, maybe the model will inheriently learn uncertainty)