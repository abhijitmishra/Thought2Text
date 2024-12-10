# Program for fine tuning eeg_encoder through image embeddings and contrastive loss
# sample command:


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
        pretrained_model_name_or_path=args.model_path
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

        batched_input_ids1 = []
        batched_input_ids2 = []

        batch_data = []

        for i, exp_label in enumerate(label_string):
            #print(f"Expected label: {exp_label}")
            #print(f"Output pred: {pred_label_strings[i]}")
            data = {}
            data["Ground Truth Image"] = image_path[i]
            data["Expected object"] = exp_label
            data["Predicted object"] = pred_label_strings[i]
            batch_data.append(data)
            new_text = text.replace("<label_string>", pred_label_strings[i])
            ps = new_text.split("<image>")
            prefix = ps[0]
            suffix = ps[1]
            print("Prefix", prefix)
            print("Suffix", suffix)
            individual_input_ids1 = tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            individual_input_ids2 = tokenizer(
                suffix.strip(),
                add_special_tokens=False,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            individual_input_ids1 = individual_input_ids1.squeeze(0)
            individual_input_ids2 = individual_input_ids2.squeeze(0)
            batched_input_ids1.append(individual_input_ids1)
            batched_input_ids2.append(individual_input_ids2)

        batched_input_ids1 = torch.stack(batched_input_ids1)
        batched_input_ids2 = torch.stack(batched_input_ids2)

        output_ids, labels_gen = model.generate(
            input_ids1=batched_input_ids1,
            input_ids2=batched_input_ids2,
            mm_embeds=emb_out,
            do_sample = False,
            max_new_tokens=max_len,
            repetition_penalty=1.1
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for j, output in enumerate(output_text):
            print("Output generated:", output)
            print("Expected caption:", caption_raw[j])
            batch_data[j]["Expected Caption"] = (
                caption_raw[j].replace("<s>", "").replace("</s>", "")
            )
            batch_data[j]["Generated Caption"] = output
            # print(labels_gen[j].shape)
            # print("Label gen", tokenizer.batch_decode(labels_gen[j].unsqueeze(0)))
        all_data += batch_data
    df = pd.DataFrame(all_data)
    df.to_csv(args.dest)


if __name__ == "__main__":
    main()
