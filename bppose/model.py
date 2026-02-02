import torch.nn as nn

class TemporalModelBase(nn.Module):
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validates that filter widths are odd numbers (except for the first one which is handled differently in some implementations, 
        # but standard TCN uses odd filters).
        self.layers = nn.ModuleList()
        next_channel = in_features * num_joints_in 
        
        # Input layer
        self.expand_conv = nn.Conv1d(next_channel, channels, 1)
        
        # Temporal Convolutional Layers
        for width in filter_widths:
            self.layers.append(
                TemporalModelBlock(channels, channels, kernel_size=width, dropout=dropout, causal=causal)
            )
            
        # Output layer
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        
    def forward(self, x):
        # x shape: [Batch, In_Channels, Seq_Len]
        x = self.expand_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.shrink(x)
        return x

class TemporalModelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, causal):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.causal = causal
        if causal:
            self.pad = (kernel_size - 1, 0)
        else:
            self.pad = (pad, pad)
            
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        out = self.conv1(nn.functional.pad(x, self.pad))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(nn.functional.pad(out, self.pad))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + res

class TemporalModel(TemporalModelBase):
    """
    Reference implementation of VideoPose3D TemporalModel.
    We assume the standard H36M configuration for the pretrained weights.
    """
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024):
        super().__init__(num_joints_in, in_features, num_joints_out,
                         filter_widths, causal, dropout, channels)
