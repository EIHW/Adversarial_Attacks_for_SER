# Adversarial_Attacks_for_SER
Pytorch code of the ICASSP 2020 paper "Generating and Protecting Against Adversarial Attacks for Deep Speech-based Emotion Recognition Models", by Zhao Ren, Alice Baird, Jing Han, Zixing Zhang, Björn Schuller.

# Data and Task
Database： the Database of Elicited Mood in Speech (DEMoS) 
Task: seven-class classification

# Preparation
channels:
  - pytorch
dependencies:
  - matplotlib=2.2.2
  - numpy=1.14.5
  - h5py=2.8.0
  - pytorch=0.4.0
  - pip:
    - audioread==2.1.6
    - librosa==0.6.1
    - scikit-learn==0.19.1
    - soundfile==0.10.2
    
# Run 
sh runme.sh

In runme.sh, please run the following files for different tasks:
1. feature extraction: utils/features.py
2. training a model, and evaluation: main_pytorch.py
  - the folder 'pytorch' is corresponding to vanilla adversarial Training
  - the folder 'pytorch-similarity' is corresponding to Similarity-based Adversarial Training
  - Please revise the '$BACKEND' to the folder name 'pytorch' or 'pytorch-similarity' in runme.sh, regarding the method which is achieved
  
# Cite
If the user referred the code, please cite our paper,

@inproceedings{ren2020generating,
title     =   {{Generating and protecting against adversarial attacks for deep speech-based emotion recognition models}},
author    =   {Ren, Zhao and Baird, Alice and Han, Jing and Zhang, Zixing and Schuller, Bj{\"o}rn},
address   =   {Barcelona, Spain},
Booktitle =   {Proc.\ ICASSP},
Year      =   {2020},
pages     =   {7184--7188}
}





Zhao Ren

ZD.B chair of Embedded Intelligence for Health Care and Wellbeing

University of Augsburg

06.07.2020
