export PYTHONPATH=.
DEVICE=0;
CONFIG="./configs/exp/durflex_evc.yaml";
SRC_WAV="./sample/0011_000021.wav"
SAVE_DIR="./results"
CUDA_VISIBLE_DEVICES=$ python infer.py --config $CONFIG \
    --src_wav $SRC_WAV \
    --save_dir $SAVE_DIR

