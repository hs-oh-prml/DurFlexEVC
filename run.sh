export PYTHONPATH=.

DEVICE=0;
CONFIG="egs/datasets/audio/esd/durflex.yaml";
MODEL_NAME="0116_durflex";

CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG --exp_name $MODEL_NAME --reset
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG --exp_name $MODEL_NAME --infer 
