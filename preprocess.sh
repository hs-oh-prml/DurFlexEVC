export PYTHONPATH=.
DEVICE=0;
CONFIG="egs/datasets/audio/esd/durflex.yaml";
python data_gen/tts/runs/preprocess.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/tts/runs/binarize.py --config $CONFIG
