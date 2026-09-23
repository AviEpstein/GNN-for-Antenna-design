import torch.nn as nn


class ResNet50(nn.Module):
    """
    Inputs:
      img16: (B,C,16,16)
    Output:
      (B,out_channels,34,34)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Use torchvision's ResNet-50 as the backbone
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        # Modify the first conv layer to accept in_channels
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the fully connected layer and avgpool
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        # Final head to map from 2048 features to out_channels * 34 * 34
        self.head = nn.Linear(2048, out_channels * 34 * 34)

    def forward(self, img16):
        assert img16.dim() == 4 and img16.shape[2:] == (16, 16), "img must be (B,C,16,16)"
        z = self.backbone(img16)      # (B,2048)
        out = self.head(z)           # (B,out_channels*34*34)
        out = out.view(img16.size(0), -1, 34, 34)  # (B,out_channels,34,34)
        return out


class MLPBaseline(nn.Module):
    def __init__(self, in_features=16*16*2, hidden=2048, out_features=34*34):
        super(MLPBaseline, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)  # Flatten the image
        out = self.mlp(x)  # (B, 34*34)
        out = out.view(b, 1, 34, 34)  # Reshape to image format
        return out
