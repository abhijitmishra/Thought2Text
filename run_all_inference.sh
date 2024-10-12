#!/bin/bash

# Declare array for LLM models
#llms=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
llms=("$1")
mkdir -p results
# Loop over each LLM model
for llm in "${llms[@]}"; do
  # Extract the second half of the LLM model for the output directory
  llm_name=$(echo "$llm" | awk -F '/' '{print $2}')

  # Construct the output directory name
  model_path="all_models/${llm_name}_all"
  model_path1="all_models/${llm_name}_no_stage2_all"

  # Run the Python command
  results_csv="results/results_${llm_name}_all.csv"
  results_csv1="results/results_${llm_name}_no_stage2_all.csv"
        
  # Execute the python inference command
  python inference.py --model_path "$model_path" \
                      --eeg_dataset data/block/eeg_55_95_std.pth \
                      --image_dir data/images/ \
                      --dest "$results_csv" \
                      --splits_path data/block/block_splits_by_image_all.pth
  python inference.py --model_path "$model_path1" \
                      --eeg_dataset data/block/eeg_55_95_std.pth \
                      --image_dir data/images/ \
                      --dest "$results_csv1" \
                      --splits_path data/block/block_splits_by_image_all.pth
  
  # Check if the command executed successfully
  if [ $? -ne 0 ]; then
    echo "Error with LLM $llm"
    exit 1
  fi
done
