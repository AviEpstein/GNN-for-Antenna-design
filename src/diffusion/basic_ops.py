import torch
from torch import nn


class Conv(nn.Module):
    """
    A convolutional layer that doesn’t change the image
    resolution, only the channel dimension
    Applies nn.Conv2d(3, 1, 1) followed by BN and GELU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the Conv layer
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H, W) output tensor.
        """
        return self.gelu(self.bn(self.conv(x)))


class DownConv(nn.Module):
    """
        A convolutional layer downsamples the tensor by 2.
        The layer consists of Conv2D(3, 2, 1) followed by BN and GELU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the DownConv layer
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H/2, W/2) output tensor.
        """
        return self.gelu(self.bn(self.conv(x)))


class UpConv(nn.Module):
    """
    A convolutional layer that upsamples the tensor by 2.
    The layer consists of ConvTranspose2d(4, 2, 1) followed by
    BN and GELU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the UpConv layer
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H*2, W*2) output tensor.
        """
        return self.gelu(self.bn(self.conv(x)))


class Flatten(nn.Module):
    """
    Average pooling layer that flattens a 7x7 tensor into a 1x1 tensor.
    The layer consists of AvgPool followed by GELU.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, 7, 7) input tensor.

        Returns:
            (N, C, 1, 1) output tensor.
        """
        return self.gelu(self.avgpool(x))


class Unflatten(nn.Module):
    """
      Convolutional layer that unflattens/upsamples a 1x1 tensor into a
      7x7 tensor. The layer consists of ConvTranspose2D(7, 7, 0)
      followed by BN and GELU.
    """
    def __init__(self, in_channels: int, kernel_size: int = 7, stride: int = 7):
        """
        Initializes Unflatten layer
        Args:
            in_channels (int): The number of input channels
        """
        super().__init__()
        self.unflatten = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, 1, 1) input tensor.

        Returns:
            (N, in_channels, 7, 7) output tensor.
        """
        return self.gelu(self.unflatten(x))

class FC(nn.Module):
    """
    Fully connected layer, consisting of nn.linear followed by GELU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the FC layer
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.gelu = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) input tensor.

        Returns:
            (N, out_channels) output tensor.
        """
        return self.gelu(self.linear(x))
