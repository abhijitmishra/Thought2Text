import random
import torch
import numpy as np
from tqdm import tqdm
from datautils import EEGDataset, Splitter
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from args import get_args_for_encoder_training
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


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()
    set_seed(42)

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
    test_loader = loaders["test"]

    config = EEGModelConfig()

    model = ChannelNetModel.from_pretrained(
        pretrained_model_name_or_path=args.output, config=config
    )
    model.to(args.device)
    model.eval()
    metric = evaluate.load("accuracy")
    softmax = torch.nn.Softmax(dim=1)
    all_labels = []
    all_preds = []
    for batch in tqdm(test_loader):
        image_raw, eeg_data, labels = batch
        image_raw = image_raw.to(args.device)
        eeg_data = eeg_data.to(args.device)
        labels = labels.to(args.device)

        emb_output, cls_output = model(eeg_data)
        preds = softmax(cls_output).argmax(dim=1)
        for l in labels:
            all_labels.append(l.item())
        for o in preds:
            all_preds.append(o.item())
    test_metric = metric.compute(predictions=all_preds, references=all_labels)
    print({"acc": test_metric["accuracy"]})


if __name__ == "__main__":
    main()
