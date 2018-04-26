# Copyright 2018 Thomas Mensink, University of Amsterdam, thomas.mensink@uva.nl
#
# Beloning to the DeepNCM repository
# DeepNCM is proposed in
#    Samantha Guerriero, Barbara Caputo, and Thomas Mensink
#    DeepNCM: Deep Nearest Class Mean Classifiers
#    ICLR Workshop 2018
#    https://openreview.net/forum?id=rkPLZ4JPM
#
# This file runs the experiments. Making uses of Parallel
# Tange, GNU Parallel - The Command-Line Power Tool, 2011
#
# Define an experiment run
doexp() {
  x=$1
  dataset=$(echo "$x" | cut -f 1 -d ";" | xargs)
  method=$(echo "$x" | cut -f 2 -d ";" | xargs)
  param=$(echo "$x" | cut -f 3 -d ";" | xargs)
  lr=$(echo "$x" | cut -f 4 -d ";" | xargs)
  logfile="logs/${dataset}_${method}_${param}_${lr}.log"
  cmd="python cifar10_deepncm.py --dataset ${dataset} --ncmmethod ${method} --ncmparam ${param} -l ${lr} >> ${logfile} 2>&1"
  echo ${cmd}
  rm ${logfile}
  eval ${cmd}
}
export -f doexp
# parallel -P ## (4) inidcates number of parallel calls:
cat deepncm_experiments.txt | parallel -P 4 doexp
