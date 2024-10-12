#!/bin/bash

# List of subject IDs
subjects=(1 2 3 4 5 6)

# List of LLMs
#llms=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
llms=("$1")

# Loop over each LLM
echo -e "Normal models"
for llm in "${llms[@]}"; do
      llm_name=$(echo "$llm" | awk -F '/' '{print $2}')
    # Loop over each subject ID
    for subject in "${subjects[@]}"; do
        # Define model path and results CSV based on the LLM and subject ID
        model_path="subject_wise/${llm_name}_subject-${subject}"
        results_csv="results/results_${llm_name}_subject-${subject}.csv"
        
        # Execute the python inference command
        python inference.py --model_path "$model_path" \
                            --eeg_dataset data/block/eeg_55_95_std.pth \
                            --image_dir data/images/ \
                            --dest "$results_csv" \
                            --splits_path data/block/block_splits_by_image_single.pth \
                            --subject "$subject"
    done
done

echo -e "No-stage2 models"

for llm in "${llms[@]}"; do
    llm_name=$(echo "$llm" | awk -F '/' '{print $2}')
    # Loop over each subject ID
    for subject in "${subjects[@]}"; do
        # Define model path and results CSV based on the LLM and subject ID
        model_path="subject_wise/${llm_name}_no_stage2-subject-${subject}"
        echo $model_path
        results_csv="results/results_${llm_name}_no_stage2-subject-${subject}.csv"
        
        # Execute the python inference command
        python inference.py --model_path "$model_path" \
                            --eeg_dataset data/block/eeg_55_95_std.pth \
                            --image_dir data/images/ \
                            --dest "$results_csv" \
                            --splits_path data/block/block_splits_by_image_single.pth \
                            --subject "$subject"
    done
done