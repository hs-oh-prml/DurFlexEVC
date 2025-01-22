
# DurFlex-EVC: Duration-Flexible Emotional Voice Conversion Leveraging Discrete Representations without Text Alignment 

<div align="center">

  <img src="assets/logo.png" alt="logo" width="400" height="auto" />  
</div>


<p align="center">
  <a href="https://github.com/hs-oh-prml/DurFlexEVC/network/members">
    <img src="https://img.shields.io/github/forks/hs-oh-prml/DurFlexEVC" alt="forks" />
  </a>
  <a href="https://github.com/hs-oh-prml/DurFlexEVC/stargazers">
    <img src="https://img.shields.io/github/stars/hs-oh-prml/DurFlexEVC" alt="stars" />
  </a>
  <br>
  <a href="https://ieeexplore.ieee.org/abstract/document/10844546">
    [Paper]
  </a>
  <a href="https://prml-lab-speech-team.github.io/durflex/">
    [Demo]
  </a>
</p>

This repository contains the implementation of DurFlex-EVC, a novel method for emotional voice conversion that flexibly handles variable durations in speech without requiring text alignment. This implementation is based on our paper:

> H.-S. Oh, S.-H. Lee, D.-H. Cho and S.-W. Lee, "DurFlex-EVC: Duration-Flexible Emotional Voice Conversion Leveraging Discrete Representations without Text Alignment," *IEEE Transactions on Affective Computing*, 2025.


## Abstract
Emotional voice conversion (EVC) involves modifying various acoustic characteristics, such as pitch and spectral envelope, to match a desired emotional state while preserving the speaker's identity. Existing EVC methods often rely on text transcriptions or time-alignment information and struggle to handle varying speech durations effectively. In this paper, we propose DurFlex-EVC, a duration-flexible EVC framework that operates without the need for text or alignment information. We introduce a unit aligner that models contextual information by aligning speech with discrete units representing content, eliminating the need for text or speech-text alignment. Additionally, we design a style autoencoder that effectively disentangles content and emotional style, allowing precise manipulation of the emotional characteristics of the speech. We further enhance emotional expressiveness through a hierarchical stylize encoder that applies the target emotional style at multiple hierarchical levels, refining the stylization process to improve the naturalness and expressiveness of the converted speech. Experimental results from subjective and objective evaluations demonstrate that our approach outperforms baseline models, effectively handling duration variability and enhancing emotional expressiveness in the converted speech.

## Getting Started

### Requirements 
```
pip install -r requirements.txt
```
Dataset: [[Download](https://hltsingapore.github.io/ESD/)]

k-means model: [[Download](https://works.do/xIux3D2)]

### Preprocessing
```
bash preprocess.sh

export PYTHONPATH=.
DEVICE=0;
CONFIG="configs/exp/durflex_evc.yaml";
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/runs/preprocess.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python data_gen/runs/binarize.py --config $CONFIG
CUDA_VISIBLE_DEVICES=$DEVICE python preprocess_unit.py --config $CONFIG

```

### Training model
```
bash run.sh

export PYTHONPATH=.
DEVICE=0;
CONFIG="configs/exp/durflex_evc.yaml";
MODEL_NAME="DurFlex";
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
    --config $CONFIG \
    --exp_name $MODEL_NAME \
    --reset

```

### Inference

```
bash infer.sh

export PYTHONPATH=.
DEVICE=0;
CONFIG="./configs/exp/durflex_evc.yaml";
SRC_WAV="./sample/0011_000021.wav"
SAVE_DIR="./results"
CUDA_VISIBLE_DEVICES=$DEVICE python infer.py --config $CONFIG \
    --src_wav $SRC_WAV \
    --save_dir $SAVE_DIR
```

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

### Citations
If you use this implementation in your research, please cite our paper:
```
@ARTICLE{10844546,
  author={Oh, Hyung-Seok and Lee, Sang-Hoon and Cho, Deok-Hyeon and Lee, Seong-Whan},
  journal={IEEE Transactions on Affective Computing}, 
  title={DurFlex-EVC: Duration-Flexible Emotional Voice Conversion Leveraging Discrete Representations Without Text Alignment}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Feature extraction;Autoencoders;Context modeling;Transformers;Acoustics;Speech recognition;Computational modeling;Vocoders;Translation;Generators;Duration control;emotional voice conversion;self-supervised representation;style disentanglement},
  doi={10.1109/TAFFC.2025.3530920}}
```