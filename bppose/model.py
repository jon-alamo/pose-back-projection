import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    """
    TemporalModel definition matching the checkpoint structure:
    - expand_conv (kernel=3, no bias) + expand_bn
    - Flat lists of layers (layers_conv, layers_bn) handling residual blocks
    - shrink conv
    """
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        super().__init__()
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.causal = causal
        self.dropout = dropout
        self.channels = channels

        # Expand Layer: 34 input channels -> 1024 channels. Kernel 3.
        # Padding 1 is used to maintain temporal dimension.
        # bias=False because the checkpoint does not have it and relies on BN.
        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, 3, padding=1, bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        
        self.layers_conv = nn.ModuleList()
        self.layers_bn = nn.ModuleList()
        
        # Generate layers for each block
        for width in filter_widths:
            # Each block has 2 convolutions
            # Conv 1
            self.layers_conv.append(nn.Conv1d(channels, channels, width, bias=False))
            self.layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            # Conv 2
            # Based on previous error "size mismatch ... copying shape (1024, 1024, 1)", the second conv is kernel 1
            self.layers_conv.append(nn.Conv1d(channels, channels, 1, bias=False))
            self.layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
        # Shrink Layer
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [Batch, Input_Channels, Seq_Len]
        
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        conv_idx = 0
        for width in self.filter_widths:
            res = x
            
            pad = (width - 1) // 2
            
            # Conv 1
            fn_pad = (width - 1, 0) if self.causal else (pad, pad)
            out = self.layers_conv[conv_idx](nn.functional.pad(x, fn_pad))
            out = self.layers_bn[conv_idx](out)
            out = self.relu(out)
            out = self.dropout_layer(out)
            conv_idx += 1
            
            # Conv 2
            # Kernel size is 1, so no padding is needed
            out = self.layers_conv[conv_idx](out)
            out = self.layers_bn[conv_idx](out)
            out = self.relu(out)
            out = self.dropout_layer(out)
            conv_idx += 1
            
            x = out + res
            
        x = self.shrink(x)
        return x
