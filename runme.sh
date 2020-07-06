#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/renzhao/demos/demos4"

# You need to modify this path as your workspace
WORKSPACE="/home/renzhao/demos/demos4/pub_demos_cnn"

DEV_SUBTASK_A_DIR="demos_data"

BACKEND="pytorch"
HOLDOUT_FOLD=1
GPU_ID=0
FEATURE="logmel"
CONVLAYER=1 

############ Extract features ############
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE

epsilon_value=(0.02 0.04 0.06 0.08 0.1)
alpha_value=(0.1)

for i in {0..4}
do
	for j in {0..0}
	do

	############ Development subtask A ############
	# Train model for subtask A
	python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --validation --holdout_fold=$HOLDOUT_FOLD --epsilon_value=${epsilon_value[$i]}  --alpha_value=${alpha_value[$j]} --cuda

	# Evaluate subtask A
	python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --validation --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --epsilon_value=${epsilon_value[$i]}  --alpha_value=${alpha_value[$j]} --cuda


	############ Full train subtask A ############
	# Train on full development data
	python $BACKEND/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --epsilon_value=${epsilon_value[$i]}  --alpha_value=${alpha_value[$j]} --cuda
	
	# Inference evaluation data
	python $BACKEND/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --epsilon_value=${epsilon_value[$i]}  --alpha_value=${alpha_value[$j]} --cuda


	done
done
