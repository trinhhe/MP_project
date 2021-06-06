# 1. TASK DESCRIPTION
In this project, we approach 3D reconstruction of animatable humans from images by leveraging SMPL - a statistical body model.

SMPL is directly controlled by human pose (per-joint 3 DoF axis-angle) and human shape coeefficients (10 DoF latent code) that your model needs to predict in order to reconstruct the human mesh geometry.

The human pose comes as skeletal representations, i.e. the human body as a set of its major joints which is organized in a tree-like, hierarchical kinematics chain that defines how the joints are linked together. Each joint is parameterized as a rotation that defines how one bone segment must be rotated to obtain the orientation of the following bone segment.

# TASK
You are given a set of 2D images that capture humans performing different actions and your task is to predict the SMPL parameters. The performance of your model is evaluated by the standard mean squared error between your predicted mesh vertices and the ground truth mesh vertices.

Along with the data, you receive a skeleton code that loads the data, trains a dummy neural network, and exports the predictions to a file that can be uploaded to the submission system. This code is only meant as a guideline; you are free to modify it to whatever extent you deem necessary.

A debugging tip is to directly visualize reconstructed 3D human geometry or to project 3D mesh vertices onto the image plane (this interface is already implemented).

# LITERATURE
To understand the problem setting and the data more in-depth, we encourage you to read some related papers. Here is a list of relevant papers where you might find some inspiration for your submission. This list is not complete and you might find other papers out there. If you're not sure whether a paper is relevant, feel free to ping your TA.

- Looper et al. (2015) SMPL: A Skinned Multi-Person Linear Model.
- Choutas et al. (2020) Monocular Expressive Body Regression through Body-Driven Attention
- Aksan and Kaufmann et al. (2019) Structured Prediction Helps 3D Human Motion Modelling 
  
Training ensembles is a well-known strategy to boost results in ML competitions. We expect you to solve the task related problems and/or **design more effective models rather than blindly using ensembles. Hence, you are not allowed to apply ensemble learning strategies** naively. More specifically, training a number of models first and then aggregating their outputs by means of majority voting or logit averaging is not permitted. We also kindly ask you not to submit any results achieved with ensemble learning techniques as they will be misleading for the others.

# DATA DESCRIPTION
The data we use is **a subset taken from Human3.6M**. It is stored directly on the cluster under`/cluster/project/infk/hilliges/lectures/mp21/project3`.
The dataset consists of **RGB images and numpy files (containing SMPL-related information) of 7 subjects. Subjects 9 and 11 are used as a test set and remaning subjects (1, 5, 6, 7, 8) as a training set.**

The training data files contain the following variables:

- betas: Shape coefficients (10-dimensional vector).
- root_loc: Root location of a human body (3-dimensional vector).
- root_orient: Root orientation of a human body (3-dimensional vector).
- pose_body: Per-joint axis-angle (21*3-dimensional vector, where 21 indicates the number of joints).
- pose_hand: Axis-angles for two hands (2*3-dimensional vector).
- scale_img and center_img: These variables define a bounding box around a captured human.

*Your task is to predict betas, root_orient, pose_hand, and pose_body on the test subjects.*  
**The provided data can only be used for this lecture.** Do not distribute it or use it for other purposes than the completion of this project.  
**You may not train on the test split (subjects 9 and 11) or any other additional data.  
However, you are allowed to use pretrained models available in PyTorch. Other pretrained models are not permitted.**  
Your final submission must be reproducible (within a reasonable range) on the data provided through this project. To this end, we will re-run your final submission. If the results are not reproducible, you may be heavily penalized.

# ENVIRONMENT SETUP
The following explains how to set up the environment and run the skeleton code on Leonhard.

## LEONHARD LOGIN
Login to Leonhard cluster:
```
ssh your_eth_username@login.leonhard.ethz.ch
Load the required CUDA modules:
module load cuda/10.1.243 cudnn/7.6.4
```

## SKELETON CODE
Copy the skeleton code via
```cp -r /cluster/project/infk/hilliges/lectures/mp21/project3/project3_skeleton``` .
It is recommended that you first read through the skeleton code to understand how it works and how you can extend it for your own purposes. Each source file is commented. It is probably a good idea to start with `train.py`.

 The skeleton code is copyright-protected. You are free to use and modify it. However, do not share your modified version with other groups or make it publicly available.

## GITLAB REPO
Your group should have received access to a Gitlab repository. Please use this repository to manage your code and submit the final report. If you haven't received a link yet, please contact your TA. To push the skeleton code into your repository for the first time, use
```
cd project3_skeleton
git init
git remote add origin git@gitlab.inf.ethz.ch:COURSE-MP2021/your_groups_name.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

## INSTALLING DEPENDENCIES
It is best to use virtual conda environment to manage your python projects. Due to quota restrictions on your $HOME directory, feel free to use $SCRATCH space.

Install Anaconda environment:
```
cd $SCRATCH
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# accept the licence and specify a desired location (e.g. $SCRATCH/conda_root)
source ~/.bashrc
```

Go to your project folder (the one we unzipped above) and run the following to create the virtual environment:
```
cd ~/project3_skeleton
conda env create -f environment.yml
```
Activate this environment:  
`conda activate mp_project3`

# SAMPLE SUBMISSION
You should now be ready to test the skeleton code. To do so, we submit a GPU job. A GPU job is a job request, which is placed in a queue and executed once a GPU is free. Hence, to avoid congestion towards the deadline, we highly encourage you to start working on the project as early as possible.

Leonhard is a shared resource, so please be considerate in its use. This means for example to request as much memory as your job requires, but not an overly excessive amount that your program does not end up using. Also, you are only allowed to use 1 GPU at a time per group.

To submit a GPU job we use the following command (make sure your virtual environment is activated at this point):
```
cd codebase/
bsub -n 6 -W 2:00 -o sample_test -R "rusage[mem=2048, ngpus_excl_p=1]" python train.py ../configs/default.yaml
```

To generate predictions you can run the following command:
```
bsub -n 6 -W 2:00 -o sample_test -R "rusage[mem=2048, ngpus_excl_p=1]" python test.py ../configs/default.yaml
```

Explanation of options:
```
-n 1 Request this many CPU cores (here 6).
-W 2:00 Wall time (here 2 hours). After running for this many hours your job is automatically cancelled. This also determines in which queue your job lands.
-o sample_test After completion of your script, its output is written into this file stored in the current directory.
-R Requested resources. mem is in MB and per requested CPU core, i.e. here we are requesting 6 GB. ngpus_excl_p is the amount of GPUs requested (must always be 1).
../configs/default.yaml
```

A configuration file for your experiment.
After you submitted the job, you can check its status via `bjobs or bbjos` (better bjobs). 
When the job is waiting in the queue its status is PEND, once it is running it will change to RUN. 
Here you can also find a unique job id. You can check some details of the job via `bjobs -l job_id` (e.g. how much memory your job used). 
Use `bpeek job_id` to peek at what the job recently wrote to standard out (note: this is buffered output and not real-time). 
Once the job finished you should find a file sample_test in the directory where you submitted the job from. 
Note that this file is always provided, no matter whether the job finished correctly or not. I.e., if your job was killed because of an exception, you can find the stack trace in this file.

These are the most important commands to monitor your jobs. More details can be found on the Leonhard wiki page.

# SUBMITTING TO THE LEADERBOARD
## SUBMISSION FILE
If test run as explained above completed successfully, you should find a new directory under `$MP_EXPERIMENTS`, which is named something like `"1617367241-DummyModel-lr0.001"`. The first 10 digits are the model's ID. This folder should contain a file `predictions_in120_out24.csv.gz`. This is the file that you can upload to the submission system. It should achieve a similar score to the existing "sample baseline". This means everything works as expected and you are now ready to work on your own model!

If you can't find this file, the training script most likely did not finish correctly. Check the output file of your job for any anomalies. You can also generate the predictions again by running `evaluate.py --model_id 1617367241`.

# EVALUATION METRIC
The evaluation metric for this project is MSE l2 between predicted and ground truth mesh vertices. Have a look at training.py to see how it is computed.

# FINAL SUBMISSION AND GRADING
In the list of your team's submission you can select which submission should be the final one. Under "3. Hand in Final Solution" you can then complete the process. Please note that each team member must make a final submission.

We provide you with one test set for which you have to compute predictions. We have partitioned this test set into two parts and use it to compute a public and a private score for each submission. You only receive feedback about your performance on the public part in the form of the public score, while the private leaderboard remains secret. The purpose of this division is to prevent overfitting to the public score. Your model should generalize well to the private part of the test set.

When handing in the task, the form asks for a short description of your approach. As we ask you the write a short report anyway (see below), you can keep this description short and simple or just copy-paste the abstract. For example, you can type "50 layer fancy resnet model for this awesome project". We will then compare your selected submission to two baselines (easy and hard). As we want to encourage novel solutions and ideas, we also ask you to write a short report (3 pages, excluding references) detailing the model you used for the final submission and your contributions. We will provide you with a report template, which will be announced separately. Depending on the report, we will weigh the grade determined by your performance w.r.t. the baselines. Hence, your final grade depends on the public score and the private score (weighted equally), on your submitted code and on your report. In addition to the report, please also include a brief README in your repo that details how to reproduce the results of your final submission.

## In summary, to complete the final submission you should:

Select the final submission in the submission system and complete step "3. Hand in Final Solution".
Push all your code required to reproduce your final submission to your group's gitlab repo, which we provided in the beginning. 
Please also include a README detailing how to re-produce your results (simply adding the command lines to train and evaluate your model with the final state of your repo is enough). We will take a snapshot of your repository at the time of the deadline. Everything pushed after the deadline will be automatically discarded.
Write and submit a brief 3-page report as explained above. You can push the PDF to your repo.
Please do not commit your trained models, intermediate data representations or any other large files. Your submitted code should be able to reproduce them.
In addition, the following non-binding guidance provides you with an idea on what is expected to obtain a certain grade: If you hand in a properly-written description, your source code is runnable and reproduces your predictions, and your submission performs better than the easy baseline, you may expect a grade exceeding a 4. If your submission performs equal to or better than the hard baseline, you may expect a 5.5. The top 2 submissions will achieve 6 and everything in-between the second best submission and the hard baseline will be linearly interpolated. The grade computed as mentioned above can go up or down, depending on the contributions of your project, as indicated by the report. If you passed the easy baseline, however, your final grade cannot go lower than 4 after the weighting.

Make sure that you properly hand in the required materials, otherwise you may obtain zero points.

# FREQUENTLY ASKED QUESTIONS
##ARE WE ALLOWED TO USE PYTORCH ONLY?
No, you can use other libraries like TensorFlow. However, support from TAs will be limited for other frameworks.

##CAN WE USE PUBLICLY AVAILABLE CODE?
Yes, you can use publicly available code. But you should specify the source as a comment in your code and your report. If your contributions are merely downloading and running existing code, you may be penalized.

##WILL YOU CHECK / RUN MY CODE?
We will check your code and compare it with other submissions. At the end of the project, we also try to reproduce your selected submission. Please make sure that your code is runnable by executing a one-line command (in addition to any packages installable via pip). Additionally, make sure your predictions are reproducible by fixing the seeds etc. Provide a readme if necessary (e.g., for installing additional libraries).

##CAN YOU GIVE ME A HINT?
We cannot give you any hints. However, you may dicuss your ideas with TAs and get feedback.

##CAN YOU GIVE ME A DEADLINE EXTENSION?
We cannot give you any extension.

##CAN I POST ON PIAZZA AS SOON AS HAVE A QUESTION?
Please follow the instructions first:

Read the details of the task thoroughly.  
Review the frequently asked questions.  
Discuss it with your team-mate.  
Try to Google it.  
If you still consider that you should contact the TAs and if it is a general question (i.e., Leonhard access, bugs, etc.) you can post on Piazza, 
if it is related with your solution and should be private you can post a private question on Piazza.  

##WHEN WILL I RECEIVE THE PRIVATE SCORES? AND THE PROJECT GRADES?
We will publish the private scores, and corresponding grades before the exam the latest.