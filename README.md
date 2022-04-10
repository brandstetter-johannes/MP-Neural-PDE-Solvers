# Message Passing Neural PDE Solvers

Johannes Brandstetter*, Daniel Worrall*, Max Welling

<a href="https://arxiv.org/abs/2202.03376">Link to the paper</a>

ICLR 2022 Spotlight Paper

If you find our work and/or our code useful, please cite us via:

```bibtex
@article{brandstetter2022message,
  title={Message Passing Neural PDE Solvers},
  author={Brandstetter, Johannes and Worrall, Daniel and Welling, Max},
  journal={arXiv preprint arXiv:2202.03376},
  year={2022}
}
```

<img src="assets/MP-PDE-Solver.png" width="800">

<img src="assets/shock_formation.png" width="800">

### Set up conda environment

source environment.sh

### Produce datasets for tasks E1, E2, E3, WE1, WE2, WE3
`python generate/generate_data.py --experiment={E1, E2, E3, WE1, WE2, WE3} --train_samples=2048 --valid_samples=128 --test_samples=128 --log=True --device=cuda:0`

###  Train MP-PDE solvers for tasks E1, E2, E3

`python experiments/train.py --device=cuda:0 --experiment={E1, E2, E3} --model={GNN, ResCNN, Res1DCNN} --base_resolution=250,{100,50,40} --time_window=25 --log=True`

### Train MP-PDE solvers for tasks WE1, WE2

`python experiments/train.py --device=cuda:0 --experiment={WE1, WE2} --base_resolution=250,{100,50,40} --neighbors=6 --time_window=25 --log=True`

### Train MP-PDE solvers for task WE3

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,100 --neighbors=20 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,50 --neighbors=12 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=10 --time_window=25 --log=True`

`python experiments/train.py --device=cuda:0 --experiment=WE3 --base_resolution=250,40 --neighbors=6 --time_window=25 --log=True`

