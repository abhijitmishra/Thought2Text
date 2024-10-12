from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from config import EEGEncoderConfig, EEGModelForCausalLMConfig
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import os

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


class EEGModelForCausalLM(PreTrainedModel):
    config_class = EEGModelForCausalLMConfig
    base_model_prefix = "eegllm"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        eeg_encoder: Optional[PreTrainedModel] = None,
        llm: Optional[PreTrainedModel] = None,
        use_lora=False,
    ):
        if config is None and (eeg_encoder is None or llm is None):
            raise ValueError(
                "Either a configuration or an eeg_encoder and an LLM model has to be provided."
            )
        if config is None:
            config = EEGModelForCausalLMConfig.from_separate_configs(
                eeg_encoder.config, llm.config
            )

        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )
        super().__init__(config)
        if eeg_encoder is None:
            eeg_encoder = ChannelNetModel(config=config.eeg_encoder)

        if llm is None:
            llm = AutoModelForCausalLM.from_config(
                config.llm,
                attn_implementation=config._attn_implementation,
            )

        self.eeg_encoder = eeg_encoder
        self.llm = llm
        self.padding_token_id = self.llm.config.eos_token_id
        self.bos_token_id = self.llm.config.bos_token_id
        self.use_lora = use_lora

        if self.eeg_encoder.config.to_dict() != self.config.eeg_encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.llm.config.to_dict() != self.config.llm.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        self.eeg_encoder.config = self.config.eeg_encoder
        self.llm.config = self.config.llm

        if self.eeg_encoder.config.embedding_size != self.llm.config.hidden_size:
            self.mm_proj = nn.Linear(
                self.eeg_encoder.config.embedding_size,
                self.llm.config.hidden_size,
            )
        else:
            self.mm_proj = nn.Linear(
                self.eeg_encoder.config.embedding_size,
                self.eeg_encoder.config.embedding_size,
            )

    def get_eeg_encoder(self):
        return self.eeg_encoder

    def get_llm(self):
        return self.llm

    def save_pretrained(self, output_dir, *model_args, **kwargs):
        # we need to save all the models separately
        
        self.eeg_encoder.save_pretrained(
            os.path.join(output_dir, "eeg_encoder"), *model_args, **kwargs
        )
        # self.llm.save_pretrained(os.path.join(output_dir, "llm"), *model_args, **kwargs)
        torch.save(
            self.mm_proj.state_dict(),
            os.path.join(output_dir, "projector.pth"),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        use_lora=False,
        *model_args,
        **kwargs,
    ):
        # TODO: Implement Download From Hub Functionality
        eeg_encoder_path = os.path.join(pretrained_model_name_or_path, "eeg_encoder")
        projector_path = os.path.join(pretrained_model_name_or_path, "projector.pth")
        llm_path = os.path.join(pretrained_model_name_or_path, "llm")
        if use_lora:
            model = None
        else:
            model = cls.from_separate_pretrained(
                eeg_encoder_path=eeg_encoder_path,
                llm_path=llm_path,
                *model_args,
                **kwargs,
            )

        model.mm_proj.load_state_dict(torch.load(projector_path))
        return model

    @classmethod
    def from_separate_pretrained(
        cls,
        eeg_encoder_path: str = None,
        llm_path: str = None,
        use_lora=False,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        kwargs_eeg_encoder = {
            argument[len("eeg_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("eeg_encoder_")
        }

        kwargs_llm = {
            argument[len("llm_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("llm_")
        }
        for key in kwargs_eeg_encoder.keys():
            del kwargs["eeg_encoder_" + key]
        for key in kwargs_llm.keys():
            del kwargs["llm_" + key]

        eeg_encoder = kwargs_eeg_encoder.pop("model", None)

        if eeg_encoder is None:
            if eeg_encoder_path is None:
                raise ValueError(
                    "If `eeg_encoder_model` is not defined as an argument, a `eeg_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_eeg_encoder:
                eeg_encoder_config, kwargs_eeg_encoder = (
                    EEGEncoderConfig.from_pretrained(
                        eeg_encoder_path,
                        **kwargs_eeg_encoder,
                        return_unused_kwargs=True,
                    )
                )

                kwargs_eeg_encoder["config"] = eeg_encoder_config

            eeg_encoder = ChannelNetModel.from_pretrained(
                eeg_encoder_path, *model_args, **kwargs_eeg_encoder
            )
        llm = kwargs_llm.pop("model", None)
        if llm is None:
            if llm_path is None:
                raise ValueError(
                    "If `llm` is not defined as an argument, a `llm_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_llm:
                llm_config, kwargs_llm = AutoConfig.from_pretrained(
                    llm_path,
                    **kwargs_llm,
                    return_unused_kwargs=True,
                )

                kwargs_llm["config"] = llm_config

            llm = AutoModelForCausalLM.from_pretrained(
                llm_path, device_map="auto", **kwargs_llm
            )

        if use_lora:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
            )
            llm = get_peft_model(llm, peft_config)
            llm.print_trainable_parameters()
        config = EEGModelForCausalLMConfig.from_separate_configs(
            eeg_encoder_config=eeg_encoder.config, llm_config=llm.config, **kwargs
        )
        return cls(eeg_encoder=eeg_encoder, llm=llm, config=config, use_lora=use_lora)

    def prepare_inputs(self, input_ids1, input_ids2, mm_emb, type="train"):
        batch_size, max_length = input_ids1.shape
        hidden_dim = mm_emb.shape[-1]
        mm_seq_len = mm_emb.shape[1]

        # Compute token embeddings
        if self.use_lora:
            try:
                input_embeds1 = self.llm.model.model.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.model.embed_tokens(input_ids2)
            except:
                # for decoder only models like OPT
                input_embeds1 = self.llm.model.model.decoder.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.model.decoder.embed_tokens(input_ids2)
        else:
            try:
                input_embeds1 = self.llm.model.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.embed_tokens(input_ids2)
            except:
                input_embeds1 = self.llm.model.decoder.embed_tokens(input_ids1)
                input_embeds2 = self.llm.model.decoder.embed_tokens(input_ids2)

        # Create attention masks (1 for non-padding tokens, 0 for padding tokens)
        attention_masks1 = (input_ids1 != self.padding_token_id).float()
        attention_masks2 = (input_ids2 != self.padding_token_id).float()

        # Compute the effective length for each input
        effective_lengths1 = attention_masks1.sum(dim=1).long()
        effective_lengths2 = attention_masks2.sum(dim=1).long()

        # Calculate the maximum effective lengths for positioning
        max_effective_length1 = max(effective_lengths1).item()
        max_effective_length2 = max(effective_lengths2).item()

        # Initialize final embeddings and labels
        final_max_length = mm_seq_len + max_effective_length1 + max_effective_length2

        final_input_embeds = torch.zeros(
            batch_size, final_max_length, hidden_dim, device=input_embeds1.device
        )
        attention_masks = torch.zeros(
            batch_size, final_max_length, device=input_embeds1.device
        )
        labels = torch.full(
            (batch_size, final_max_length),
            IGNORE_INDEX,
            device=input_ids1.device,
        )

        for i in range(batch_size):
            effective_length1 = effective_lengths1[i].item()
            effective_length2 = effective_lengths2[i].item()

            total_len = mm_seq_len + effective_length1 + effective_length2
            start_idx = final_max_length - total_len
            
            final_input_embeds[
                i,
                start_idx: start_idx + effective_length1,
                :,
            ] = input_embeds1[i, -effective_length1:, :]

            final_input_embeds[i, start_idx+ effective_length1 : start_idx + effective_length1+ mm_seq_len, :] = mm_emb[
                i, :, :
            ]

            final_input_embeds[
                i, start_idx + effective_length1 + mm_seq_len : final_max_length, :
            ] = input_embeds2[i, -effective_length2:, :]

            attention_masks[i, start_idx:] = 1
            labels[
                i, start_idx: start_idx + effective_length1
            ] = input_ids1[i, -effective_length1:]
            
            # labels[i, start_idx+ effective_length1 : start_idx + effective_length1+ mm_seq_len] = IGNORE_INDEX
            

            labels[i, start_idx + effective_length1 + mm_seq_len : final_max_length] = (
                input_ids2[i, -effective_length2:]
            )

        # Create position ids
        final_input_embeds = final_input_embeds.to(input_embeds1.dtype)

        # attention_masks = attention_masks.to(input_embeds1.dtype)
        # print(attention_masks)
        if type != "train":
            attention_masks = None

        return final_input_embeds, attention_masks, labels

    def forward(
        self,
        input_ids1,
        input_ids2,
        mm_embeds=None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        mm_embeds = self.mm_proj(mm_embeds)

        if len(mm_embeds.shape) == 2:
            # We are working on pooled embeddings now, but in the future, patched embeddings are possible
            # prepare_inputs assumes a sequence of mm_embeds, hence a shape of B*S*N
            # Pooled embeddings : B *N -> B*S*N -> B*1*N for now
            mm_embeds = mm_embeds.unsqueeze(1)
        final_input_embeds, attention_masks, labels = self.prepare_inputs(
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            mm_emb=mm_embeds,
        )

        # with torch.no_grad():
        try:
            llm_outputs = self.llm(
                input_ids=None,
                attention_mask=attention_masks,
                inputs_embeds=final_input_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs,
            )
        except:
            # decoder only models like OPT
            llm_outputs = self.llm(
                input_ids=None,
                inputs_embeds=final_input_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs,
            )

        return llm_outputs, labels

    def generate(
        self,
        input_ids1,
        input_ids2,
        mm_embeds=None,
        **kwargs,
    ):
        mm_embeds = self.mm_proj(mm_embeds)
        if len(mm_embeds.shape) == 2:
            # We are working on pooled embeddings now, but in the future, patched embeddings are possible
            # prepare_inputs assumes a sequence of mm_embeds, hence a shape of B*S*N
            # Pooled embeddings : B *N -> B*S*N -> B*1*N for now
            mm_embeds = mm_embeds.unsqueeze(1)
        # Switch this on if you want to test without projector
        # mm_embeds = torch.zeros_like(mm_embeds).to(mm_embeds.device)

        final_input_embeds, attention_masks, labels = self.prepare_inputs(
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            mm_emb=mm_embeds,
            type="inference",
        )
        output_ids = self.llm.generate(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=final_input_embeds,
            **kwargs,
        )
        return output_ids, labels
