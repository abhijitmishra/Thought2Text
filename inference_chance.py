# this is to generate text by chance
# python inference_wo_stage1.py --model_path mistral7b-eeg_55_95_40_classes --splits_path data/block/block_splits_by_image_all.pth --image_dir data/images/ --eeg_dataset data/block/eeg_55_95_std.pth --dest results/mistral_55_95_40_classes_results_CHANCE.csv

import random
import logging
import torch
import json
import os
import numpy as np


from tqdm import tqdm
from args import get_args_for_llm_inference
from model import EEGModelForCausalLM
from datautils import EEGInferenceDataset, SplitterInference
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
import pandas as pd


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


def main():
    set_seed(42)
    args = get_args_for_llm_inference()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if "gemma" in args.model_path.lower():
        messages = [
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
    else:
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    max_len = 100

    print("Loading model...")

    model = EEGModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
    )

    # For stage 3, we only train the mm_projector, everything else is static
    model.eeg_encoder.to(args.device)
    model.mm_proj.to(args.device)
    model.eval()
    softmax = torch.nn.Softmax(dim=1)

    dataset = EEGInferenceDataset(
        args=args,
    )
    loaders = {
        split: DataLoader(
            SplitterInference(
                dataset,
                split_path=args.splits_path,
                split_num=args.split_num,
                split_name=split,
            ),
            batch_size=1,
            drop_last=True,
            shuffle=True,
        )
        for split in ["train", "val", "test"]
    }
    test_dataloader = loaders["test"]

    with open(os.path.join(args.model_path, "id2label.json")) as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    all_data = []

    for batch in tqdm(test_dataloader):
        eeg, label_string, caption_raw, image_path = batch
        eeg = eeg.to(args.device)
        emb_out, cls_out = model.eeg_encoder(eeg)
        preds = softmax(cls_out).argmax(dim=1)

        pred_label_strings = []
        for p in preds:
            pred_label_strings.append(id2label[p.item()])

        batch_data = []
        text_data = []

        for i, exp_label in enumerate(label_string):
            # print(f"Expected label: {exp_label}")
            # print(f"Output pred: {pred_label_strings[i]}")
            data = {}
            data["Ground Truth Image"] = image_path[i]
            data["Expected object"] = exp_label
            data["Predicted object"] = pred_label_strings[i]
            batch_data.append(data)
            new_text = text.replace("<label_string>", pred_label_strings[i])
            #ps = new_text.split("<image>")
            text_data.append(new_text)

        batched_input_ids = tokenizer(
            text_data,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        #print("BBB", batched_input_ids.shape)

        output_ids = model.llm.generate(
            input_ids=batched_input_ids,
            max_new_tokens=max_len,
            repetition_penalty=1.1
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for j, output in enumerate(output_text):
            # print("Output generated:", output)
            # print("Expected caption:", caption_raw[j])
            batch_data[j]["Expected Caption"] = (
                caption_raw[j].replace("<s>", "").replace("</s>", "")
            )
            batch_data[j]["Generated Caption"] = (
                output.split(pred_label_strings[j], 1)[1]
                .replace("This image depicts", "")
                .replace("The image shows", "")
                .replace("This image shows", "")
                .replace("The image depicts", "")
                .strip()
            )  # replace repeatition of pro
            # print(labels_gen[j].shape)
            # print("Label gen", tokenizer.batch_decode(labels_gen[j].unsqueeze(0)))
        # print(batch_data)
        all_data += batch_data
    df = pd.DataFrame(all_data)
    df.to_csv(args.dest)


if __name__ == "__main__":
    main()
