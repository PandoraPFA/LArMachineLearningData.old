# model.py

import torch.nn as nn
import torch


def maxpool():
    """Return a max pooling layer.

        The maxpooling layer has a kernel size of 2, a stride of 2 and no padding.

        Returns:
            The max pooling layer
    """
    return nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)


def dropout(prob):
    """Return a dropout layer.

        Args:
            prob: The probability that drop out will be applied.

        Returns:
            The dropout layer
    """
    return nn.Dropout(prob)


def reinit_layer(seq_block, leak = 0.0, use_kaiming_normal=True):
    """Reinitialises convolutional layer weights.

        The default Kaiming initialisation in PyTorch is not optimal, this method
        reinitialises the layers using better parameters

        Args:
            seq_block: The layer to be reinitialised.
            leak: The leakiness of ReLU (default: 0.0)
            use_kaiming_normal: Use Kaiming normal if True, Kaiming uniform otherwise (default: True)
    """
    for layer in seq_block:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            if use_kaiming_normal:
                nn.init.kaiming_normal_(layer.weight, a = leak)
            else:
                nn.init.kaiming_uniform_(layer.weight, a = leak)
                layer.bias.data.zero_()


class ConvBlock(nn.Module):
    """A convolution block
    """

    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        """Constructor.

            Args:
                c_in: The number of input channels
                c_out: The number of output channels
                k_size: The size of the convolution filter
                k_pad: The amount of padding around the images
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = k_size, padding = k_pad, stride = 1),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size = k_size, padding = k_pad, stride = 1),
            nn.GroupNorm(8, c_out)
        )
        reinit_layer(self.block)

    def forward(self, x):
        """Forward pass.

            Args:
                x: The input to the layer

            Returns:
                The output from the layer
        """
        return self.block(x)

class TransposeConvBlock(nn.Module):
    """A tranpose convolution block
    """

    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        """Constructor.

            Args:
                c_in: The number of input channels
                c_out: The number of output channels
                k_size: The size of the convolution filter
                k_pad: The amount of padding around the images
        """
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size = k_size, padding = k_pad, output_padding = 1, stride = 2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True))
        reinit_layer(self.block)

    def forward(self, x):
        """Forward pass.

            Args:
                x: The input to the layer

            Returns:
                The output from the layer
        """
        return self.block(x)

class Sigmoid(nn.Module):
    """A sigmoid activation function that supports categorical cross-entropy
    """

    def __init__(self, out_range = None):
        """Constructor.

            Args:
                out_range: A tuple covering the minimum and maximum values to map to
        """
        super(Sigmoid, self).__init__()
        if out_range is not None:
            self.low, self.high = out_range
            self.range = self.high - self.low
        else:
            self.low = None
            self.high = None
            self.range = None

    def forward(self, x):
        """Applies the sigmoid function.

            Rescales to the specified range if provided during construction

            Args:
                x: The input to the layer

            Returns:
                The (potentially scaled) sigmoid of the input
        """
        if self.low is not None:
            return torch.sigmoid(x) * (self.range) + self.low
        else:
            return torch.sigmoid(x)

class ListModule(nn.Module):
    """A container for a list of modules.

        This class provides flexibility for the network architecture by ensuring layers in a configurable
        architecture are correctly registered with torch.nn.Module
    """

    def __init__(self, *args):
        """Constructor.

            Args:
                args: A list of modules to be added to the network
        """
        super(ListModule, self).__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def __getitem__(self, idx):
        """Retrieve a module.

            Args:
                idx: The index of the module to be retrieved

            Returns:
                The requested module
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """Retrieve an iterator for the modules

            Returns:
                An iterator for the modules
        """
        return iter(self._modules.values())

    def __len__(self):
        """Retrieve the number of modules

            Returns:
                The number of modules
        """
        return len(self._modules)

class UNet(nn.Module):
    """A U-Net for semantic segmentation.
    """

    def __init__(self, in_dim, n_classes, depth = 4, n_filters = 16, drop_prob = 0.1, y_range = None):
        """Constructor.

            Args:
                in_dim: The number of input channels
                n_classes: The number of classes
                depth: The number of convolution blocks in the downsampling and upsampling arms of the U (default: 4)
                n_filters: The number of filters in the first layer (doubles for each downsample) (default: 16)
                drop_prob: The dropout probability for each layer (default: 0.1)
                y_range: The range of values (low, high) to map to in the output (default: None)
        """
        super(UNet, self).__init__()
        # Contracting Path
        ds_convs = []
        for i in range(depth):
            if i == 0: ds_convs.append(ConvBlock(in_dim, n_filters * 2**i))
            else: ds_convs.append(ConvBlock(n_filters * 2**(i - 1), n_filters * 2**i))
        self.ds_convs = ListModule(*ds_convs)

        ds_maxpools = []
        for i in range(depth):
            ds_maxpools.append(maxpool())
        self.ds_maxpools = ListModule(*ds_maxpools)

        ds_dropouts = []
        for i in range(depth):
            ds_dropouts.append(dropout(drop_prob))
        self.ds_dropouts = ListModule(*ds_dropouts)

        self.bridge = ConvBlock(n_filters * 2**(depth - 1), n_filters * 2**depth)

        # Expansive Path
        us_tconvs = []
        for i in range(depth, 0, -1):
            us_tconvs.append(TransposeConvBlock(n_filters * 2**i, n_filters * 2**(i - 1)))
        self.us_tconvs = ListModule(*us_tconvs)

        us_convs = []
        for i in range(depth, 0, -1):
            us_convs.append(ConvBlock(n_filters * 2**i, n_filters * 2**(i - 1)))
        self.us_convs = ListModule(*us_convs)

        us_dropouts = []
        for i in range(depth):
            us_dropouts.append(dropout(drop_prob))
        self.us_dropouts = ListModule(*us_dropouts)

        self.output = nn.Sequential(nn.Conv2d(n_filters * 1, n_classes, 1), Sigmoid(y_range))

    def forward(self, x):
        """Forward pass.

            Args:
                x: The input to the layer

            Returns:
                The output from the layer
        """
        res = x
        conv_stack = []

        # Downsample
        for i in range(len(self.ds_convs)):
            res = self.ds_convs[i](res); conv_stack.append(res)
            res = self.ds_maxpools[i](res)
            res = self.ds_dropouts[i](res)

        # Bridge
        res = self.bridge(res)

        # Upsample
        for i in range(len(self.us_convs)):
            res = self.us_tconvs[i](res)
            res = torch.cat([res, conv_stack.pop()], dim=1)
            res = self.us_dropouts[i](res)
            res = self.us_convs[i](res)

        output = self.output(res)

        return output