"""
Temporal convolutional model for 3D pose estimation.
Based on VideoPose3D from Facebook Research.
"""

import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    Uses dilated convolutions to achieve large receptive field efficiently.
    """
    
    def __init__(
        self, 
        num_joints_in: int, 
        in_features: int, 
        num_joints_out: int,
        filter_widths: list, 
        causal: bool = False, 
        dropout: float = 0.25, 
        channels: int = 1024
    ):
        """
        Initialize the temporal model.
        
        Args:
            num_joints_in: Number of input joints (e.g. 17 for COCO).
            in_features: Number of input features per joint (2 for x,y coordinates).
            num_joints_out: Number of output joints.
            filter_widths: List of convolution widths, determines blocks and receptive field.
            causal: Use causal convolutions for real-time applications.
            dropout: Dropout probability.
            channels: Number of convolution channels.
        """
        super().__init__()
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.causal = causal
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Padding for each layer
        self.pad = [filter_widths[0] // 2]
        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        
        # Initial expansion convolution
        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, 
            channels, 
            filter_widths[0], 
            bias=False
        )
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        
        # Build dilated convolution layers
        layers_conv = []
        layers_bn = []
        
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)
            
            # Dilated convolution
            layers_conv.append(nn.Conv1d(
                channels, 
                channels,
                filter_widths[i],
                dilation=next_dilation,
                bias=False
            ))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            # 1x1 convolution
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
        # Output shrink layer
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        
    def receptive_field(self) -> int:
        """
        Calculate the total receptive field of this model in frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, frames, joints, features).
            
        Returns:
            Output tensor of shape (batch, output_frames, joints, 3).
        """
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        batch_size = x.shape[0]
        
        # Reshape: (batch, frames, joints, features) -> (batch, joints*features, frames)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Expansion block
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Residual blocks with dilated convolutions
        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            
            # Extract residual (center portion that matches output size)
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            # First conv (dilated)
            x = self.layers_conv[2 * i](x)
            x = self.layers_bn[2 * i](x)
            x = self.relu(x)
            x = self.drop(x)
            
            # Second conv (1x1) + residual
            x = self.layers_conv[2 * i + 1](x)
            x = self.layers_bn[2 * i + 1](x)
            x = self.relu(x)
            x = self.drop(x)
            
            x = x + res
        
        # Output
        x = self.shrink(x)
        
        # Reshape: (batch, joints*3, frames) -> (batch, frames, joints, 3)
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, -1, self.num_joints_out, 3)
        
        return x
