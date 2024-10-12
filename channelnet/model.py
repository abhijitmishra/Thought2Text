# This is the model presented in the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by
# Multimodal Learning of Neural Activity and Visual Features,  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
import torch
import torch.nn as nn
from .config import EEGModelConfig
from transformers import PreTrainedModel

from .layers import *


class FeaturesExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.temporal_block = TemporalBlock(
            config.in_channels,
            config.temp_channels,
            config.num_temp_layers,
            config.temporal_kernel,
            config.temporal_stride,
            config.temporal_dilation_list,
            config.input_width,
        )

        self.spatial_block = SpatialBlock(
            config.temp_channels * config.num_temp_layers,
            config.out_channels,
            config.num_spatial_layers,
            config.spatial_stride,
            config.input_height,
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                    ),
                    ConvLayer2D(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                        config.down_kernel,
                        config.down_stride,
                        0,
                        1,
                    ),
                )
                for i in range(config.num_residual_blocks)
            ]
        )

        self.final_conv = ConvLayer2D(
            config.out_channels * config.num_spatial_layers,
            config.out_channels,
            config.down_kernel,
            1,
            0,
            1,
        )

    def forward(self, x):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)

        return out


class ChannelNetModel(PreTrainedModel):
    """The model for EEG classification.
    The imput is a tensor where each row is a channel the recorded signal and each colums is a time sample.
    The model performs different 2D to extract temporal e spatial information.
    The output is a vector of classes where the maximum value is the predicted class.
    Args:
        in_channels: number of input channels
        temp_channels: number of features of temporal block
        out_channels: number of features before classification
        num_classes: number possible classes
        embedding_size: size of the embedding vector
        input_width: width of the input tensor (necessary to compute classifier input size)
        input_height: height of the input tensor (necessary to compute classifier input size)
        temporal_dilation_list: list of dilations for temporal convolutions, second term must be even
        temporal_kernel: size of the temporal kernel, second term must be even (default: (1, 32))
        temporal_stride: size of the temporal stride, control temporal output size (default: (1, 2))
        num_temp_layers: number of temporal block layers
        num_spatial_layers: number of spatial layers
        spatial_stride: size of the spatial stride
        num_residual_blocks: the number of residual blocks
        down_kernel: size of the bottleneck kernel
        down_stride: size of the bottleneck stride
    """

    def __init__(self, config: EEGModelConfig):
        super().__init__(config=config)

        self.encoder = FeaturesExtractor(config=config)

        encoding_size = (
            self.encoder(
                torch.zeros(
                    1, config.in_channels, config.input_height, config.input_width
                )
            )
            .contiguous()
            .view(-1)
            .size()[0]
        )
        self.projector = nn.Linear(encoding_size, config.embedding_size)
        self.classifier = nn.Linear(config.embedding_size, config.num_classes)

    def forward(self, x):
        out = self.encoder(x)

        out = out.view(x.size(0), -1)
        emb = self.projector(out)

        cls = self.classifier(emb)

        return emb, cls
