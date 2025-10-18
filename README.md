# cdred
Official code implementation for paper: Coupled Distributional Random Expert Distillation for World Model Online Imitation Learning

# CDRED-WM
Official code implementation for paper: Coupled Distributional Random Expert Distillation for World Model Online Imitation Learning[[Paper Link]](https://arxiv.org/abs/2505.02228)

[Shangzhe Li](https://tobyleelsz.github.io/), [Zhiao Huang](https://sites.google.com/view/zhiao-huang), [Hao Su](https://cseweb.ucsd.edu/~haosu/)

##  Introduction

CDRED-WM is a world model that introduces a novel approach to world model-based online imitation learning through an innovative reward model formulation. CDRED-WM grounds its reward model in density estimation over both expert and behavioral state-action distributions with a coupled RND-based estimator. Built upon the architecture of [TD-MPC2](https://www.tdmpc2.com/), CDRED-WM excels in handling diverse tasks while maintaining consistent stability during long-term onling training.

## Environment Setup and Running the Code

1. Setup the environment using the following commands:
```
conda env create -f conda_env/environment.yaml
conda activate cdred
```
2. Download the expert datasets [here](https://drive.google.com/drive/folders/1-D5tDFIjhta2cFq44BTEtW4mptcBr502?usp=sharing), which includes the expert datasets for 5 locomotion tasks. All of the expert demonstrations are sampled from a trained single-task TD-MPC2 agent. The expert datasets for manipulation tasks will be updated soon.
3. Set the task in cdred_wm/config.json and the correct expert dataset path corresponding to the task.
4. Run the training code:
```
python3 cdred_wm/train.py
```
## Acknowledgement

This repository is created based on the original TD-MPC2 implementation repository: [TD-MPC2 Official Implementation](https://github.com/nicklashansen/tdmpc2).

## Citation

If you find our work helpful to your research, please consider citing our paper as follows:
```
@article{li2025coupled,
  title={Coupled Distributional Random Expert Distillation for World Model Online Imitation Learning},
  author={Li, Shangzhe and Huang, Zhiao and Su, Hao},
  journal={arXiv preprint arXiv:2505.02228},
  year={2025}
}
```
