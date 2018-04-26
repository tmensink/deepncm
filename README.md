# DeepNCM: Deep Nearest Class Means
This software provides some initial DeepNCM models

## Dependencies
It is written in python, and follows (as closely as possible) the Tensorflow official ResNet implementation.
It uses GNU Parallel for experiments (Tange, GNU Parallel - The Command-Line Power Tool, 2011)

## Citation
When using this code, or the ideas of DeepNCM, please cite the following paper ([pdf](https://openreview.net/forum?id=rkPLZ4JPM))

    @INPROCEEDINGS{guerriero18openreview,
     author = {Samantha Guerriero and Barbara Caputo and Thomas Mensink},
     title = {DeepNCM: Deep Nearest Class Mean Classifiers},
     booktitle = {International Conference on Learning Representations - Workshop (ICLRw)},
     year = {2018},
     }

## ToDO
- TF Models as subgit/module?
- Jupyter Notebook
- Gradient clipping is now set to (-1.0,1.0), this could be checked

## Copyright
Source code is written by Samantha Guerriero and Thomas Mensink.
(c) 2018 University of Amsterdam, Thomas Mensink
