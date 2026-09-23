import torch
from torch import nn
from src.diffusion.basic_ops import Conv, DownConv, UpConv, FC, Flatten, Unflatten


class ConvBlock(nn.Module):
    """
    Two consecutive Conv operations.
    Note that it has the same input and output shape as Conv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes ConvBlock
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H, W) output tensor.
        """
        return self.conv2(self.conv1(x))


class DownBlock(nn.Module):
    """
    DownConv followed by ConvBlock. Note that it has the same input and output
    shape as DownConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes DownBlock
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.downconv = DownConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H/2, W/2) output tensor.
        """
        return self.conv(self.downconv(x))


class UpBlock(nn.Module):
    """
    UpConv followed by ConvBlock.
    Note that it has the same input and output shape as UpConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes UpBlock
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.upconv = UpConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels, H, W) input tensor.

        Returns:
            (N, out_channels, H*2, W*2) output tensor.
        """
        return self.conv(self.upconv(x))

class FCBlock(nn.Module):
    """
    Fully-connected Block, consisting of FC layer followed by Linear layer. Note
    that it has the same input and output shape as FC.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes FCBlock
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
        """
        super().__init__()
        self.fc = FC(in_channels, out_channels)
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) input tensor.

        Returns:
            (N, out_channels) output tensor.
        """
        return self.linear(self.fc(x))


class UNetBaselinePixelModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int
    ):
        super().__init__()
        self.conv_in = Conv(in_channels, num_hiddens)

        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)

        self.flatten = Flatten(kernel_size=4)
        self.unflatten = Unflatten(num_hiddens * 2, kernel_size=4)

        self.up1 = UpBlock(4 * num_hiddens, num_hiddens)
        self.up2 = UpBlock(2 * num_hiddens, num_hiddens)

        self.conv_out = Conv(2 * num_hiddens, num_hiddens)
        self.conv2D = nn.Conv2d(num_hiddens, in_channels, 1)

        self.fc1 = FC(9, num_hiddens)
        self.fc2 = FC(9, num_hiddens * 2)

        # Additional upsampling layer to achieve 34x34 output
        self.upsample = nn.Upsample(size=(34, 34), mode='bilinear', align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        env: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            env: (N, 1) normalized environment tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (16, 16), "Expect input shape to be (16, 16)."

        # time embedding
        enc_t_1 = self.fc1(env).unsqueeze(-1).unsqueeze(-1)               # (N, num_hiddens, 1, 1)
        enc_t_2 = self.fc2(env).unsqueeze(-1).unsqueeze(-1)               # (N, num_hiddens * 2, 1, 1)

        # encoder
        enc1 = self.conv_in(x)                                          # (N, num_hiddens, 28, 28)
        enc2 = self.down1(enc1)                                         # (N, num_hiddens, 14, 14)
        enc3 = self.down2(enc2)                                         # (N, num_hiddens * 2, 7, 7)

        z = self.flatten(enc3)                                          # (N, num_hiddens * 2, 1, 1)

        # decoder
        unflatten = self.unflatten(z)                                   # (N, num_hiddens * 2, 7, 7)

        input_to_up_1 = torch.cat((enc3, unflatten + enc_t_2), dim=1)   # (N, num_hiddens * 4, 7, 7)
        dec1 = self.up1(input_to_up_1)                                  # (N, num_hiddens , 14, 14)
        input_to_up_2 = torch.cat((enc2, dec1 + enc_t_1), dim=1)        # (N, num_hiddens * 2, 14, 14)
        dec2 = self.up2(input_to_up_2)                                  # (N, num_hiddens , 28, 28)

        input_to_out = torch.cat((enc1, dec2), dim=1)                   # (N, num_hiddens * 2, 28, 28)
        dec3 = self.conv_out(input_to_out)                              # (N, num_hiddens , 28, 28)

        output = self.conv2D(dec3)                                      # (N, in_channels, 28, 28)

        # Upsample to 34x34
        return self.upsample(output)


class UNetBaselineBasicModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_hiddens: int
    ):
        super().__init__()
        self.conv_in = Conv(in_channels, num_hiddens)

        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)

        self.flatten = Flatten(kernel_size=4)
        self.unflatten = Unflatten(num_hiddens * 2, kernel_size=4)

        self.up1 = UpBlock(4 * num_hiddens, num_hiddens)
        self.up2 = UpBlock(2 * num_hiddens, num_hiddens)

        self.conv_out = Conv(2 * num_hiddens, num_hiddens)
        self.conv2D = nn.Conv2d(num_hiddens, out_channels, 1)
        # Additional upsampling layer to achieve 34x34 output
        self.upsample = nn.Upsample(size=(34, 34), mode='bilinear', align_corners=False)


    def forward(
        self,
        x: torch.Tensor,
        env = None
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (16, 16), "Expect input shape to be (16, 16)."


        # encoder
        enc1 = self.conv_in(x)                                          # (N, num_hiddens, 28, 28)
        enc2 = self.down1(enc1)                                         # (N, num_hiddens, 14, 14)
        enc3 = self.down2(enc2)                                         # (N, num_hiddens * 2, 7, 7)

        z = self.flatten(enc3)                                          # (N, num_hiddens * 2, 1, 1)

        # decoder
        unflatten = self.unflatten(z)                                   # (N, num_hiddens * 2, 7, 7)

        input_to_up_1 = torch.cat((enc3, unflatten), dim=1)   # (N, num_hiddens * 4, 7, 7)
        dec1 = self.up1(input_to_up_1)                                  # (N, num_hiddens , 14, 14)
        input_to_up_2 = torch.cat((enc2, dec1), dim=1)        # (N, num_hiddens * 2, 14, 14)
        dec2 = self.up2(input_to_up_2)                                  # (N, num_hiddens , 28, 28)
        input_to_out = torch.cat((enc1, dec2), dim=1)                   # (N, num_hiddens * 2, 28, 28)
        dec3 = self.conv_out(input_to_out)                              # (N, num_hiddens , 28, 28)

        output = self.conv2D(dec3)                                      # (N, in_channels, 28, 28)

        # Upsample to 34x34
        return self.upsample(output)
