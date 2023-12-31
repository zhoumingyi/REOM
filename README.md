# Code157
Code for the submission 157

The dependency can be found in `environment.yml`. To create the conda environment:

`conda env create -f environment.yml`

`conda activate code157`

We randomly sampled 64 images for testing the accuracy and attack success rate.

Note that the `acc_mode=True` refers the REOM disables some pruning rules that transform the quantized output to float output to ensure the dibuggability. If we keep such pruning rules, the output scale will be changed so that we cannot compare the transformation error between the converted model and the source model.

To evaluate the scaled transformation error:

`python tflite2pytorch.py --all --save_onnx --acc_mode`

To evaluate the accuracy and the attack success of the converted model:

`
bash attack.sh
`

The above command will run the evaluation for all models. To evaluate a specific source model, you can first convert the TFLite model using our method (without the --acc_mode):

`
python tflite2pytorch.py --model_name=bird --save_onnx
`

Then, you can evaluate the robustness of the source model using the reverse-engineered model:

`
python attack.py --cuda --adv=BIM --model=bird --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a attack.txt
`

Note that if you want to evaluate the performance in the white-box setting, you can add the '--white-box' config to enable the white-box evaluation. For testing the model using the black-box setting, we recommend using Foolbox (https://github.com/bethgelab/foolbox) to apply the black-box algorithms to the source models. 
