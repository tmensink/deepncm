# DeepNCM: Deep Nearest Class Means
This software provides DeepNCM models based on the TensFlow Models Official ResNet implementation.

## Citation
When using this code, or the ideas of DeepNCM, please cite the following paper ([openreview](https://openreview.net/forum?id=rkPLZ4JPM))

    @INPROCEEDINGS{guerriero18openreview,
     author = {Samantha Guerriero and Barbara Caputo and Thomas Mensink},
     title = {DeepNCM: Deep Nearest Class Mean Classifiers},
     booktitle = {International Conference on Learning Representations - Workshop (ICLRw)},
     year = {2018},
     }

### Dependencies
It is written in python, and follows (as closely as possible) the Tensorflow official ResNet implementation.
  - Requires TensorFlow
  - Requires TensorFlow Models
    - Included as submodule, so after cloning this library do  
    `git submodule update --init`
  - The experiments (deepncm_do.sh) uses GNU Parallel for parallelisation (Tange, GNU Parallel - The Command-Line Power Tool, 2011)


## Experimental overview on Cifar10/Cifar100
Below are the full experiments, using two learning rates, different condensation (omreset) and decay rates.
![DeepNCM Experimental Overiew](https://github.com/tmensink/deepncm/blob/master/figs/exp_cifar_overview.png)
Comparison of the following methods: Softmax (sof), Online Means (onl), Mean Condensation (con), Decay Mean (dec), in the legend the maximum Top-1 accuracy is reported.

# Future research (ideas)
- Current optimiser and learning-rate schedule is optimised for softmax learning.
- Gradient clipping is now set to (-1.0,1.0), this is not tuned
- Experiments on larger datasets, _e.g._, ImageNet
- Class incremental / Open Set learning

Please contact me when your're interested to collaborate on this

### Copyright (2017-2018)
Thomas Mensink, University of Amsterdam, thomas.mensink@uva.nl   
Some preliminary source code is written by Samantha Guerriero and Thomas Mensink.
