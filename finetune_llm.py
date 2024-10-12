# Program for fine tuning eeg_encoder through image embeddings and contrastive loss
# sample command:

# python finetune_llm.py
#   --eeg_dataset data/block/eeg_55_95_std.pth
#   --splits_path data/block/block_splits_by_image_all.pth
#   --eeg_encoder_path ./eeg_encoder_55-95_40_classes
#   --image_dir data/images/ --output mistral7b-eeg_55_95_40_classes
#   --llm_backbone_name_or_path mistralai/Mistral-7B-Instruct-v0.3
#   --load_in_8bit

# For skipping stage3:

# python finetune_llm.py --eeg_dataset data/block/eeg_55_95_std.pth --splits_path data/block/block_splits_by_image_all.pth --eeg_encoder_path ./eeg_encoder_55-95_40_classes --image_dir data/images/ --output mistral7b-eeg_55_95_40_classes_no_stage3 --llm_backbone_name_or_path mistralai/Mistral-7B-Instruct-v0.3 --load_in_8bit --no_stage3


import os
import gc
import random
import logging
import torch
import numpy as np
import json
import copy
from transformers import (
    CLIPVisionModelWithProjection,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from datautils import (
    EEGFineTuningDataset,
    SplitterFineTuning,
    Filter
)
from torch.utils.data import Dataset, DataLoader
from args import get_args_for_llm_finetuning
from model import EEGModelForCausalLM


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


class Stage2Trainer(Trainer):
    def __init__(self, clip_model=None, data_loaders=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        (
            img_data,
            eeg_data,
            input_ids1,
            input_ids2,
            label_string,
        ) = inputs
        pixels = img_data["pixel_values"].to(self.device)
        image_embeddings = self.clip_model(pixels).image_embeds
        #image_embeddings = image_embeddings.to(self.device)
        output, labels = model(
            input_ids1=input_ids1, input_ids2=input_ids2, mm_embeds=image_embeddings
        )
        # print("Labels", self.tokenizer.batch_decode(labels))
        return (output.loss, output) if return_outputs else output.loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]


class Stage3Trainer(Trainer):
    def __init__(self, data_loaders=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.data_loaders = data_loaders
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        (
            eeg_data,
            input_ids1,
            input_ids2,
        ) = inputs
        #eeg_data = eeg_data.to(self.device)
        #input_ids1 = input_ids1.to(self.device)
        #input_ids2 = input_ids2.to(self.device)
        output, labels = model(
            input_ids1=input_ids1, input_ids2=input_ids2, mm_embeds=eeg_data
        )
        # print("Labels", self.tokenizer.batch_decode(labels))
        return (output.loss, output) if return_outputs else output.loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]


def main():
    set_seed(42)
    args = get_args_for_llm_finetuning()
    dtype = torch.float32

    if args.load_in_8bit:
        logger.info("Model in INT8")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        model = EEGModelForCausalLM.from_separate_pretrained(
            eeg_encoder_path=args.eeg_encoder_path,
            llm_path=args.llm_backbone_name_or_path,
            use_lora=args.use_lora,
            llm_quantization_config=quantization_config,
            llm_low_cpu_mem_usage=True,
        )
        args.optim = "paged_adamw_8bit"
        model.eeg_encoder.to(args.device)
        model.mm_proj.to(args.device)

    else:
        logger.info("Model in FULL")
        model = EEGModelForCausalLM.from_separate_pretrained(
            eeg_encoder_path=args.eeg_encoder_path,
            llm_path=args.llm_backbone_name_or_path,
            use_lora=args.use_lora,
            llm_low_cpu_mem_usage=True,
        )
        model.eeg_encoder.to(args.device)
        model.mm_proj.to(args.device)

    model.llm.save_pretrained(os.path.join(args.output, "llm"))
    model.train()
    set_gradients(module=model.eeg_encoder, requires_grad=False)
    set_gradients(module=model.llm, requires_grad=False)

    dataset = EEGFineTuningDataset(
        args=args, tokenizer_path=args.llm_backbone_name_or_path
    )
    
    if not args.no_stage2:
        logger.info("STAGE 2: LLM fine tuning on images")
        llm_name = args.llm_backbone_name_or_path.split("/")[1]
        pretrained_path = os.path.join(args.saved_pretrained_model_path, llm_name)
        
        if os.path.exists(pretrained_path) and os.path.isdir(pretrained_path):
            print(f"Stage 3 trained model already available. Loadig model from {pretrained_path}. Skipping retraining")
            del model
            gc.collect()
            model = EEGModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,llm_low_cpu_mem_usage= True
            )

            model.eeg_encoder.to(args.device)
            model.mm_proj.to(args.device)
            set_gradients(module=model.eeg_encoder, requires_grad=False)
            model.llm.save_pretrained(os.path.join(pretrained_path, "llm"))


        else:           
            training_arguments_stage2 = TrainingArguments(
                output_dir=args.output,
                num_train_epochs=args.num_epochs_image,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=True,
                optim=args.optim,
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                fp16=args.fp16,
                bf16=args.bf16,
                max_grad_norm=args.max_grad_norm,
                max_steps=args.max_steps,
                warmup_ratio=args.warmup_ratio,
                group_by_length=args.group_by_length,
                lr_scheduler_type=args.lr_scheduler_type,
                report_to="tensorboard",
            )


            # Load CLIP model for stage 3
            clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model)
            clip_model.requires_grad_(False)
            clip_model.eval()
            clip_model.to(args.device)
            if (args.subject!=0):
                # for subjectwise analysis
                # We need to warmup with all images
                new_args = copy.deepcopy(args)
                new_args.subject = 0
                new_args.splits_path = new_args.splits_path.replace("image_single", "image_all")
                img_dataset = EEGFineTuningDataset(
                    args=new_args, tokenizer_path=args.llm_backbone_name_or_path
                )   
                loaders = {
                    split: DataLoader(
                        SplitterFineTuning(
                            img_dataset,
                            split_path=new_args.splits_path,
                            split_num=new_args.split_num,
                            split_name=split,
                        ),
                        batch_size=new_args.batch_size,
                        drop_last=True,
                        shuffle=True,
                    )
                    for split in ["train", "val", "test"]
                }
                trainer = Stage2Trainer(
                    model=model,
                    args=training_arguments_stage2,
                    train_dataset=img_dataset,
                    eval_dataset=img_dataset,
                    data_loaders=loaders,
                    clip_model=clip_model,
                    tokenizer=img_dataset.tokenizer,
                )
            else:
                loaders = {
                    split: DataLoader(
                        SplitterFineTuning(
                            dataset,
                            split_path=args.splits_path,
                            split_num=args.split_num,
                            split_name=split,
                        ),
                        batch_size=args.batch_size,
                        drop_last=True,
                        shuffle=True,
                    )
                    for split in ["train", "val", "test"]
                }

                trainer = Stage2Trainer(
                    model=model,
                    args=training_arguments_stage2,
                    train_dataset=dataset,
                    eval_dataset=dataset,
                    data_loaders=loaders,
                    clip_model=clip_model,
                    tokenizer=dataset.tokenizer,
                )
            trainer.train()
            model.save_pretrained(pretrained_path)
            model.llm.save_pretrained(os.path.join(pretrained_path, "llm"))
            dataset.tokenizer.save_pretrained(pretrained_path)

            del clip_model
            del loaders
            gc.collect()
    
    loaders = {
            split: DataLoader(
                Filter(SplitterFineTuning(
                    dataset,
                    split_path=args.splits_path,
                    split_num=args.split_num,
                    split_name=split,
                ),eeg_encoder = model.eeg_encoder, device = args.device),
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=True,
            )
            for split in ["train", "val", "test"]
        }
    

    training_arguments_stage3 = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs_eeg,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="tensorboard",
    )

    trainer = Stage3Trainer(
        model=model,
        args=training_arguments_stage3,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_loaders=loaders,
        tokenizer=dataset.tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output)
    dataset.tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "id2label.json"), "w") as f:
        json.dump(dataset.id2label, f)


if __name__ == "__main__":
    main()
