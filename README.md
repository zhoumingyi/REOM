# REOM
Code for the ICSE paper "Investigating White-Box Attacks for On-Device Models".

## Abstract

Numerous mobile apps have leveraged deep learning capabilities. However, on-device models are vulnerable to attacks as they can be easily extracted from their corresponding mobile apps. Although the structure and parameters information of these models can be accessed, existing on-device attacking approaches only generate black-box attacks (i.e., indirect white-box attacks), which are far less effective and efficient than white-box strategies. This is because mobile deep learning (DL) frameworks like TensorFlow Lite (TFLite) do not support gradient computing (referred to as non-debuggable models), which is necessary for white-box attacking algorithms. Thus, we argue that existing findings may underestimate the harmfulness of on-device attacks. To this end, we conduct a study to answer this research question: Can on-device models be directly attacked via white-box strategies? We first systematically an-
alyze the difficulties of transforming the on-device model to its debuggable version, and propose a Reverse Engineering framework for On-device Models (REOM), which automatically reverses the compiled on-device TFLite model to its debuggable version, enabling attackers to launch white-box attacks. Our empirical results show that our approach is effective in achieving automated transformation (i.e., 92.6%) among 244 TFLite models. Compared with previous attacks using surrogate models, REOM enables attackers to achieve higher attack success rates (10.23%→89.03%) with a hundred times smaller attack perturbations (1.0→0.01). Our findings emphasize the need for developers to carefully consider their model deployment strategies, and use white-box methods to evaluate the vulnerability of on-device models.

## Setup

We provide two ways to build the environment to test our proposed tool. If you want to build the environment from scratch, it requires you to have a workstation with Ubuntu OS. In addition, you need to install Git and Anaconda. For other users, we recommend following the guidelines in the option 1 to build the environment using Docker. However, the provided docker image does not support the GPU acceleration.

### Option 1: Build The Environment from Scratch

(1) First, you need to download the whole code and code from the GitHub using the command:

```
git clone https://github.com/zhoumingyi/REOM.git
```

(2) The dependency can be found in `environment.yml`. To create the conda environment:


```
conda env create -f environment.yml
```

(3) Then activate the created conda environment

```
conda activate reom
```

### Option 2: Build The Environment using Docker

(1) To build the environment using Docker, you need to first download the Docker image:

```
docker pull zhoumingyigege/reom:latest
```

(2) Then, enter the environment:

```
docker run -i -t zhoumingyigege/reom:latest /bin/bash
```

(3) Next, enter the project and activate the conda environment:

```
cd reom/
conda activate reom
```

## Run

### Reproduce the major results (i.e., transformation error, accuracy, attack success rate) of our paper:

(1) To evaluate the scaled transformation error:

```
python tflite2pytorch.py --all --save_onnx --acc_mode  | tee -a error.txt
```

It will convert TFLite models to PyTorch models that are saved in the 'pytorch_model' folder. It works fine when it outputs some error information because some operators are not supported by ONNX, and our method will fix it later. It will also compute the scaled max error and min error which will be logged in the 'error.txt' file. The results should be similar to the results shown in Figure 7 of our paper. Note that the 'acc_mode=True' option refers to the converted model will have the same output type and scale as the source model. We need to set it as True for comparing the transformation error between the converted model and the source model. However, for the next step (i.e., evaluating the accuracy and attack success rate), the output type should be changed to float if the source model uses UInt8 computation to guarantee the debuggability of the converted model.

(2) To evaluate the accuracy (shown in Table 5) and the attack success (shown in Table 6) of the converted model:

```
bash attack.sh
```

It will log the accuracy and attack success rate in the 'acc_asr.txt' file. The results should be similar to Table 5 and Table 6 of our paper. However, it is acceptable when the results have a small difference from the original results because we only provide 64 samples to test our method in our code repository (the original datasets contain hundreds of GB data). The sampled data can be found in the 'dataset/'. We also provide a list (https://github.com/zhoumingyi/reom/blob/main/dataset_list.txt) that contains the link to complete datasets.

### To evaluate the REOM-based attack on your own model

(1) Suppose you have a Fruit Recognition model, you can first cp your model to the 'tflite\_model' folder and convert the TFLite model to PyTorch model using our method (without the --acc_mode):

```
python tflite2pytorch.py --model_name=fruit --save_onnx
```

(2) Then, you can evaluate the robustness of the source model using the reverse-engineered model:

```
python attack.py --cuda --adv=BIM --model=fruit 
```
