#!/bin/sh

python3 precalc_fid.py --dataset celeba --dataroot '/data' --outf '/fid/celeba_fid_stats.npz'

python3 train.py --dataset celeba --homedir '/swg/' --batch_size 64 --lr 0.0001 --loss montecarlo --max_epochs 20
python3 train.py --dataset celeba --homedir '/swg/' --batch_size 64 --lr 0.0001 --loss montecarlo --max_epochs 20 --disc --crossval
python3 train.py --dataset celeba --homedir '/swg/' --batch_size 64 --lr 0.0001 --loss clt --max_epochs 20 --disc --crossval
