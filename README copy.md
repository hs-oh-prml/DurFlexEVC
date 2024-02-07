# DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation (Under review)

H.-S. Oh, S.-H. Lee, D.-H. Cho, and S.-W. Lee

![image](https://github.com/hs-oh-prml/DurFlexEVC/assets/43984708/3b13d0cf-eefb-4e3e-8b5a-09d336e3581d)

## Abstract

Emotional voice conversion (EVC) seeks to modify the emotional tone of a speaker’s voice while preserving the original linguistic content and the speaker’s unique vocal characteristics.
Recent advancements in EVC have involved the simultaneous modeling of pitch and duration, utilizing the potential of sequence-to-sequence (seq2seq) models.
To enhance reliability and efficiency in conversion, this study shifts focus towards parallel speech generation.
We introduce Duration-Flexible EVC (DurFlex-EVC), which integrates a style autoencoder and unit aligner.
Traditional models, while incorporating self-supervised learning (SSL) representations that contain both linguistic and paralinguistic information, have neglected this dual nature, leading to reduced controllability.
Addressing this issue, we implement cross-attention to synchronize these representations with various emotions.
Additionally, a style autoencoder is developed for the disentanglement and manipulation of style elements.
The efficacy of our approach is validated through both subjective and objective evaluations, establishing its superiority over existing models in the field.

## Usage

## Dataset

We use Emotional-Speech-Data (ESD) for our experiments [https://github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data)
