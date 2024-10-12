#!/bin/bash

# Declare array for LLM models
#llms=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
llms=("$1")
# Loop over each LLM model
for llm in "${llms[@]}"; do
  # Extract the second half of the LLM model for the output directory
  llm_name=$(echo "$llm" | awk -F '/' '{print $2}')

  # Construct the output directory name
  output_dir="all_models/${llm_name}_all"
  output_dir1="all_models/${llm_name}_no_stage2_all"

  # Run the Python command
  python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "$output_dir" \
    --llm_backbone_name_or_path "$llm" \
    --load_in_8bit \
    --bf16
  python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "$output_dir1" \
    --llm_backbone_name_or_path "$llm" \
    --no_stage2 \
    --load_in_8bit \
    --bf16

  # Check if the command executed successfully
  if [ $? -ne 0 ]; then
    echo "Error with LLM $llm"
    exit 1
  fi
done
