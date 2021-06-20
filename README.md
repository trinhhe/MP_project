# Team HSD submission

## Members
Daniel Vegh (davegh@student.ethz.ch),  
Husmann Severin (shusmann@student.ethz.ch)  
Henry Trinh (trinhhe@student.ethz.ch)

## Description

In this project, the goal was to conduct 3D reconstruction of animatable humans directly from 2D images by leveraging the skinned multi-person linear model (SMPL). For this reason we employ a GAN architecture on the Human 3.6M dataset.


## Setup

### Installation

Clone this repository.
```bash
$ git clone https://gitlab.inf.ethz.ch/COURSE-MP2021/HSD/
$ cd HSD
```

We suggest to create a virtual environment and install the required packages.
```bash
$ conda env create -f environment.yml
$ conda activate mp_project3
```


### Source Code Directory Tree
```
.
└── codebase                # Source code for the experiments
    ├── data                # Dataset including preprocessing functions
    ├── regressor           # Model architecture and trainer
    └── .py files           # Base files to run experiments as described below
├── configs                 # Configruation to run final experiment and SMPL mean shape
└── project_report.pdf      # Our group's project report
```
<!-- └── misc                # Images for README -->

### Codebase Structure

- `fulltraining.py`: Running the full training on all train/val subjects
- `test.py`: Prediction on test set script


## How to run on the Leonhard Cluster:
```
module load cuda/10.1.243 cudnn/7.6.4
module load eth_proxy
conda activate mp_project3
cd codebase/
```

To train the model:
```
bsub -n 6 -W 24:00 -o hsd_experiment -R "rusage[mem=2048, ngpus_excl_p=1]" python fulltraining.py --epochs 3
```

To predict:
```
bsub -n 6 -W 2:00 -o hsd_test -R "rusage[mem=2048, ngpus_excl_p=1]" python test.py --model_path /tmp_out/generator_best.pt
```
