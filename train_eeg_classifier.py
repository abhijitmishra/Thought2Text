import os
import random
import json
import torch
import numpy as np
from tqdm import tqdm
from datautils import EEGDataset, Splitter
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from args import get_args_for_encoder_training
from loss import MSELoss
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    CLIPVisionModelWithProjection,
)
from torch.utils.data import DataLoader, Dataset
import evaluate


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


class EEGEncoderTrainer(Trainer):
    def __init__(
        self,
        emb_loss_fn=None,
        cls_loss_fn=None,
        clip_model=None,
        data_loaders=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_loss_fn = emb_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.metric = evaluate.load("accuracy")
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = "cpu"

    def compute_loss(self, model, inputs, return_outputs=False):
        self.model.train()
        img_data, eeg, labels = inputs
        image_embeddings = self.clip_model(
            pixel_values=img_data["pixel_values"]
        ).image_embeds
        emb_output, cls_output = model(eeg)
        emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
        cls_loss = self.cls_loss_fn(cls_output, labels)
        loss = cls_loss + emb_loss
        self.device = eeg.device
        return (loss, cls_output) if return_outputs else loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset=None)
        eval_loss = 0
        all_labels = []
        all_preds = []
        for batch in tqdm(eval_dataloader):
            image_raw, eeg_data, labels = batch
            image_raw = image_raw.to(self.device)
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            image_embeddings = self.clip_model(
                pixel_values=image_raw["pixel_values"]
            ).image_embeds
            emb_output, cls_output = self.model(eeg_data)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            cls_loss = self.cls_loss_fn(cls_output, labels)
            loss = cls_loss + emb_loss
            eval_loss += loss.item()
            preds = self.softmax(cls_output).argmax(dim=1)
            for l in labels:
                all_labels.append(l.item())
            for o in preds:
                all_preds.append(o.item())
        eval_metric = self.metric.compute(predictions=all_preds, references=all_labels)
        print({"eval_loss": eval_loss, "acc": eval_metric["accuracy"]})

        # Do testing
        test_dataloader = self.get_test_dataloader(test_dataset=None)
        test_loss = 0
        all_labels = []
        all_preds = []
        for batch in tqdm(test_dataloader):
            image_raw, eeg_data, labels = batch
            image_raw = image_raw.to(self.device)
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            image_embeddings = self.clip_model(
                pixel_values=image_raw["pixel_values"]
            ).image_embeds
            emb_output, cls_output = self.model(eeg_data)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            cls_loss = self.cls_loss_fn(cls_output, labels)
            loss = cls_loss + emb_loss
            test_loss += loss.item()
            preds = self.softmax(cls_output).argmax(dim=1)
            for l in labels:
                all_labels.append(l.item())
            for o in preds:
                all_preds.append(o.item())
        test_metric = self.metric.compute(predictions=all_preds, references=all_labels)
        print({"test_loss": test_loss, "acc": test_metric["accuracy"]})

        return {"eval_loss": -eval_metric["accuracy"]}


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()
    set_seed(42)
    # processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model)
    clip_model.to(args.device)
    clip_model.requires_grad_(False)
    set_gradients(clip_model, False)
    clip_model.eval()

    dataset = EEGDataset(args=args)
    loaders = {
        split: DataLoader(
            Splitter(
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

    config = EEGModelConfig()

    config.save_pretrained(args.output)
    model = ChannelNetModel(config=config)

    training_arguments = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        load_best_model_at_end=True,
        save_strategy="epoch",
        eval_strategy="epoch",
    )
    trainer = EEGEncoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        emb_loss_fn=MSELoss(),
        cls_loss_fn=torch.nn.CrossEntropyLoss(),
        data_loaders=loaders,
        clip_model=clip_model,
    )
    trainer.train()
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
