export PYTHONPATH=.
DEVICE=0;
CONFIG="configs/exp/durflex_evc.yaml";
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/runs/preprocess.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/runs/binarize.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python preprocess_unit.py --config $CONFIG
