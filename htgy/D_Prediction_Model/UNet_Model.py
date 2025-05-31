import torch
import torch.nn as nn
import torch.nn.functional as F

from globalss import *

class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=2, features=[64, 128, 256, 512]):
        """
            - Input Shape: [batch_size, 8, 844, 1263]
            - in_channels:  Total = 8
                Dynamic:    SOC fast, SOC slow, V fast, V slow, Precipitation, Active check dams 
                Static:     DEM, River mask  
            - out_channels:   SOC fast, SOC slow
        """
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Downsampling Path
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        
        # Upsampling path
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature*2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # upsample
            skip = skip_connections[i // 2]
            
            # padding if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i+1](x)    # conv
            
        return self.final_conv(x)