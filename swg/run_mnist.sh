#!/bin/sh

python3 precalc_fid.py --dataset mnist --dataroot '/data' --outf '/fid/mnist_fid_stats.npz'

python3 train.py --dataset mnist --homedir '/swg/' --loss montecarlo --max_epochs 200
python3 train.py --dataset mnist --homedir '/swg/' --loss montecarlo --max_epochs 200 --disc --crossval
python3 train.py --dataset mnist --homedir '/swg/' --loss clt --max_epochs 200 --disc --crossval
