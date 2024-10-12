from transformers import PretrainedConfig


class EEGModelConfig(PretrainedConfig):
    model_type = "eeg_channelnet"

    def __init__(
        self,
        in_channels=1,
        temp_channels=10,
        out_channels=50,
        num_classes=40,
        embedding_size=512,
        input_width=440,
        input_height=128,
        temporal_dilation_list=None,
        temporal_kernel=(1, 33),
        temporal_stride=(1, 2),
        num_temp_layers=4,
        num_spatial_layers=4,
        spatial_stride=(2, 1),
        num_residual_blocks=4,
        down_kernel=3,
        down_stride=2,
        **kwargs
    ):
        if temporal_dilation_list is None:
            temporal_dilation_list = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)]

        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.temp_channels = temp_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_width = input_width
        self.input_height = input_height
        self.temporal_dilation_list = temporal_dilation_list
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        self.num_temp_layers = num_temp_layers
        self.num_spatial_layers = num_spatial_layers
        self.spatial_stride = spatial_stride
        self.num_residual_blocks = num_residual_blocks
        self.down_kernel = down_kernel
        self.down_stride = down_stride
