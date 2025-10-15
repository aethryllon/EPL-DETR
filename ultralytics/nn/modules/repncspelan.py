import torch
import torch.nn as nn

from .conv import Conv
from .block import C2f

class RELANet(nn.Module):
    """RELA Network - Replace RepNCSPELAN with this implementation"""
    
    def __init__(self, c1, c2, c3, c4, n=1, e=0.5):
        """Initializes the RELANet module with specified input/output channels and number of layers."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = Conv(c3, c3, 3, 1)
        self.cv4 = Conv(c3, c3, 3, 1)
        self.cv5 = Conv(c3, c3, 3, 1)
        self.cv6 = Conv(c3, c4, 1, 1)
        self.cv7 = Conv(c4 + c3 + c3 + c3, c2, 1, 1)
        
        # Using C2f blocks for better feature extraction
        self.m = C2f(c3, c3, n=n, shortcut=False)

    def forward(self, x):
        """Forward pass through the RELANet module."""
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y3 = self.cv3(y2)
        y4 = self.cv4(y3)
        y5 = self.cv5(y4)
        y6 = self.cv6(y5)
        y7 = self.m(y2)
        return self.cv7(torch.cat([y1, y7, y4, y6], 1))