# Thought2Text: Text Generation from EEG Signal using Large Language Models (LLMs)

Paper Link: https://arxiv.org/pdf/2410.07507v1 

**Abstract**: Decoding and expressing brain activity  in a comprehensible form is a challenging frontier in AI. This paper presents \textit{Thought2Text}, which uses instruction-tuned Large Language Models (LLMs) fine-tuned with EEG data to achieve this goal. The approach involves three stages: (1) training an EEG encoder for visual feature extraction, (2) fine-tuning LLMs on image and text data, enabling multimodal description generation, and (3) further fine-tuning on EEG embeddings to generate text directly from EEG during inference. Experiments on a public EEG dataset collected for six subjects with image stimuli demonstrate the efficacy of multimodal LLMs (LLaMa-v3, Mistral-v0.3, Qwen2.5), validated using traditional language generation evaluation metrics, GPT-4 based assessments, and evaluations by human expert. This approach marks a significant advancement towards portable, low-cost "thoughts-to-text" technology with potential applications in both neuroscience and natural language processing (NLP).

## Approach
Thought2Text implements a three stage training approach to fine tune LLMs and make them Visual EEG aware. A sneak peak can be had by looking at the following diagram (and for detailed explanation refer to Section 4 of the paper). 

![diagram](https://github.com/abhijitmishra/Thought2Text/blob/main/diagrams/method_thought_text.jpg)
    

## Sample outputs from Mistral-v0.3-Instruct

![samples](https://github.com/abhijitmishra/Thought2Text/blob/main/diagrams/examples.png)


## Data and Stage1 Pretrained Model: 
Download the data from [here](https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu?usp=sharing) and place it inside a newly created `data` directory. **Note:** We do not hold copytight on the data, except the text descriptions, the data is shared only for reproducibility and only for academic research. If you have any questions about he original data, please contact the original authors (CITATION BELOW). 

Once you have downloaded the data, install all dependencies through `pip install -r requirements.txt`

## Training 

### Stage1: EEG Encoder alignmment with CLIP embeddings

```
python train_eeg_classifier.py --eeg_dataset data/block/eeg_55_95_std.pth --splits_path data/block/block_splits_by_image_all.pth --output ./eeg_encoder_55-95_40_classes --image_dir data/images/
```
The checkpoints for the encoder will be stored in `./eeg_encoder_55-95_40_classes`. 

### Stage2 and Stage 3: Fine tuning Command

We use MistralV3 7B model as our example. Also `/path/to/encoder` should point to `output/path/to/save/encoder` from stage2. 

```
python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "mistral_eeg_model" \
    --llm_backbone_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --load_in_8bit \
    --bf16

```

Upon completion, the traine model will be available under `mistral_eeg_model` directory. 

For model variants, in-subject and cross-subjectanalysis, refer to `run.sh` which captures all commands. 

## Inference:
For inference, run `inference.py` while pointing to the fine tuned model directory and path to `eeg_55_95_std.pth` and `block_splits_by_image_all.pth`. Sample commands with MistralV3 7B, assuming that the trained model is in `mistral_eeg_model` directory is given below:

```
python inference.py \
    --model_path "$model_path" \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --image_dir data/images/ \
    --dest "mistral_results.csv"
```

## Evaluation:
We evaluate the model's generations through popular NLG metrics such as BLEU, METEOR and ROUGE. We also measure fluency and adequacy through GPT-4. The IPYNB notebooks can be found inside the `eval` folder. 

## Ethics Statement:
For this work, we utilized anonymized open-source EEG data, acknowledging the sensitivity of EEG data and the imperative of ethical compliance. All experimental data used in our research were anonymized to protect participant privacy and uphold ethical standards.

## Acknowledgement:
The EEG Encoder portion of our approach is based on the following paper and the Channelnet encoder code and EEG data is based on [this repository](https://github.com/perceivelab/eeg_visual_classification). We sincerely thank tha authors for their novel contribution which has made this work possible. 

- S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909

For any questions or concerns, contact [Abhijit](mailto:abhijitmishra.530@gmail.com) or [Shreya](mailto:shreya.shukla@utexas.edu). Pull requests and GitHub issues may not be entertained in time. If you use our work, please cite it.  
