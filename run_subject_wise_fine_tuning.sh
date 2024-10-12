#!/bin/bash

# Declare arrays for LLM models and subject numbers
#llms=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
llms=("$1")


subjects=(1 2 3 4 5 6)

# Loop over each LLM model
for llm in "${llms[@]}"; do
  # Extract the second half of the LLM model for the output directory
  llm_name=$(echo "$llm" | awk -F '/' '{print $2}')
  echo "$llm"
  # Loop over each subject number
  for subject in "${subjects[@]}"; do
    # Construct the output directory name
    output_dir="subject_wise/${llm_name}_subject-${subject}"
    output_dir1="subject_wise/${llm_name}_no_stage2-subject-${subject}"
    #Run the Python command
    python finetune_llm.py \
      --eeg_dataset data/block/eeg_55_95_std.pth \
      --splits_path data/block/block_splits_by_image_single.pth \
      --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
      --image_dir data/images/ \
      --output "$output_dir" \
      --llm_backbone_name_or_path "$llm" \
      --subject "$subject" \
      --load_in_8bit \
      --bf16
    
    python finetune_llm.py \
      --eeg_dataset data/block/eeg_55_95_std.pth \
      --splits_path data/block/block_splits_by_image_single.pth \
      --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
      --image_dir data/images/ \
      --output "$output_dir1" \
      --llm_backbone_name_or_path "$llm" \
      --subject "$subject"\
      --no_stage2 \
      --load_in_8bit \
      --bf16

    # Check if the command executed successfully
    if [ $? -ne 0 ]; then
      echo "Error with subject $subject and LLM $llm"
      exit 1
    fi
  done
done
