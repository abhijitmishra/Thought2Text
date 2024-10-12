from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


class EEGEncoderConfig(PretrainedConfig):
    model_type = "eegencoder"

    def __init__(
        self,
        d_model=64,
        attn_heads=4,
        dropout=0.2,
        reg_layers=4,
        enable_res_parameter=1,
        momentum=0.99,
        vocab_size=192,
        wave_length=8,
        mask_ratio=0.6,
        data_shape=[256, 5],
        device="cuda",
        num_class=1,
        encoder_hidden_size=64,
        projection_hidden_size=512,
        layers=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.reg_layers = reg_layers
        self.enable_res_parameter = enable_res_parameter
        self.momentum = momentum
        self.vocab_size = vocab_size
        self.wave_length = wave_length
        self.mask_ratio = mask_ratio
        self.data_shape = data_shape
        self.device = device
        self.num_class = num_class
        self.encoder_hidden_size = encoder_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.layers = layers
        return None


class EEGModelForCausalLMConfig(PretrainedConfig):
    model_type = "eeg-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "eeg_encoder" not in kwargs or "llm" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )
        eeg_encoder_config = kwargs.pop("eeg_encoder")
        llm_config = kwargs.pop("llm")
        llm_model_type = llm_config.pop("model_type")

        self.eeg_encoder = EEGEncoderConfig(**eeg_encoder_config)
        self.llm = AutoConfig.for_model(llm_model_type, **llm_config)

    @classmethod
    def from_separate_configs(
        cls,
        eeg_encoder_config: PretrainedConfig,
        llm_config: PretrainedConfig,
        **kwargs,
    ) -> PretrainedConfig:

        return cls(
            eeg_encoder=eeg_encoder_config.to_dict(), llm=llm_config.to_dict(), **kwargs
        )
