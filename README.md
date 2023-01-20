## Code for *Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections*

This repository contains the implementation of the experiments presented in <a href="https://arxiv.org/abs/2106.15427" target="_blank">Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections</a> (publication accepted at NeurIPS 2021).

### Usage

* To reproduce synthetic experiments (Figures 1 and 2): in the directory *'synthetic_exp'*, run `python3 main.py`
* To reproduce results on image generation (Table 1, Figure 3): in the directory *'swg'*, run `./run_mnist.sh` (for MNIST dataset) or `./run_celeba.sh` (for CelebA dataset)

### Citation

If you use this code in a scientific publication, please cite the following paper.

```BibTeX
@inproceedings{nadjahi2021fast,
 author = {Nadjahi, Kimia and Durmus, Alain and Jacob, Pierre E and Badeau, Roland and Simsekli, Umut},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {12411--12424},
 publisher = {Curran Associates, Inc.},
 title = {Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections},
 url = {https://proceedings.neurips.cc/paper/2021/file/6786f3c62fbf9021694f6e51cc07fe3c-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
