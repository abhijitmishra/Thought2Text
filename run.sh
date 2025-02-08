export CUDA_VISIBLE_DEVICES=0
echo "Running all fine-tuning"
bash run_all_fine_tuning.sh $1

echo "Running subject-wise fine-tuning"
bash run_subject_wise_fine_tuning.sh $1

echo "Running all inference"
bash run_all_inference.sh $1

echo "Running subject-wise inference"
bash run_subject_wise_inference.sh $1

echo "Running inference chance"
bash run_inference_chance.sh $1

echo "Running inference chance 2"
bash run_inference_chance2.sh $1

echo "Running inference without object"
bash run_inference_only_eeg.sh $1
