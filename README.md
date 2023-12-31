# REOM
Code for the ICSE paper "Investigating White-Box Attacks for On-Device Models".

## Abstract

Numerous mobile apps have leveraged deep learning capabilities. However, on-device models are vulnerable to attacks as they can be easily extracted from their corresponding mobile apps. Although the structure and parameters information of these models can be accessed, existing on-device attacking approaches only generate black-box attacks (i.e., indirect white-box attacks), which are far less effective and efficient than white-box strategies. This is because mobile deep learning (DL) frameworks like TensorFlow Lite (TFLite) do not support gradient computing (referred to as non-debuggable models), which is necessary for white-box attacking algorithms. Thus, we argue that existing findings may underestimate the harmfulness of on-device attacks. To this end, we conduct a study to answer this research question: Can on-device models be directly attacked via white-box strategies? We first systematically an-
alyze the difficulties of transforming the on-device model to its debuggable version, and propose a Reverse Engineering framework for On-device Models (REOM), which automatically reverses the compiled on-device TFLite model to its debuggable version, enabling attackers to launch white-box attacks. Our empirical results show that our approach is effective in achieving automated transformation (i.e., 92.6%) among 244 TFLite models. Compared with previous attacks using surrogate models, REOM enables attackers to achieve higher attack success rates (10.23%→89.03%) with a hundred times smaller attack perturbations (1.0→0.01). Our findings emphasize the need for developers to carefully consider their model deployment strategies, and use white-box methods to evaluate the vulnerability of on-device models.

## Setup

The dependency can be found in `environment.yml`. To create the conda environment:

```conda env create -f environment.yml```

```conda activate reom```

## Run

We randomly sampled 64 images for testing the accuracy and attack success rate. The sampled data can be found in the 'dataset/' folder.

Note that the `acc_mode=True` option refers the REOM disables some pruning rules that transform the quantized output to float output to ensure the differentiability (debuggability). If we keep such pruning rules, the output scale will be changed so that we cannot compare the transformation error between the converted model and the source model.

To evaluate the scaled transformation error:

```python tflite2pytorch.py --all --save_onnx --acc_mode```

To evaluate the accuracy and the attack success of the converted model:


```bash attack.sh```

The above command will run the evaluation for all models. To evaluate a specific source model, you can first convert the TFLite model using our method (without the --acc_mode):


```python tflite2pytorch.py --model_name=bird --save_onnx```

Then, you can evaluate the robustness of the source model using the reverse-engineered model:


```python attack.py --cuda --adv=BIM --model=bird --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a attack.txt```

Note that if you want to evaluate the performance in the white-box setting, you can add the '--white-box' config to enable the white-box evaluation. For testing the model using the black-box setting, we recommend using Foolbox (https://github.com/bethgelab/foolbox) to apply the black-box algorithms to the source models. 
