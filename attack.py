from __future__ import print_function
import argparse
import os
import gc
import sys
import onnx
import time
import random
from glob import glob
from multiprocessing import cpu_count
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from onnx_pytorch.onnx_pytorch import ConvertModel
from utils.import_tflite_model import tf_inference
from modifier.pruning import pruning
from modifier.translation import translation
from modifier.auto_matching import auto_matching
from utils.utils import *
from utils.read_data_skin import get_data, CustomDataset

num_threads = cpu_count()
if num_threads > 8:
    num_threads = 8
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--adv', type=str, default='PGD', help='attack method')
parser.add_argument('--eps', type=float, default=0.1, help='eps')
parser.add_argument('--nb_iter', type=int, default=250, help='nb_iter')
parser.add_argument('--eps_iter', type=float, default=0.0015, help='eps_iter')
parser.add_argument('--model', type=str, help='target model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='targeted attack')
parser.add_argument('--white_box', action='store_true', help='white-box attack')
parser.add_argument('--batch_size', type=int, default=64, help='Batch_size')

opt = parser.parse_args()
# print(opt)


cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you can probably run with --cuda")
    print("Set device as cuda:0")
    device = torch.device("cuda:0")
else:
    device = torch.device("cuda:0" if opt.cuda else "cpu")

def onnx_modifier(onnx_model):
    pruning(onnx_model)
    translation(onnx_model)
    auto_matching(onnx_model, similarity=0.0)

def test_acc(attack_net, target_net, dtype='uint8'):
    #-----------------------------------
    # Obtain the accuracy of the model
    #-----------------------------------
    with torch.no_grad():
        correct_att = 0.0
        correct_tar = 0.0
        total = 0.0
        attack_net.eval()
        if dtype == 'uint8':
            max_pixel = 255.0
        elif dtype == 'float32':
            max_pixel = 1.0

        inputs = torch.load(os.path.join('./dataset/',opt.model, 'inputs.pt'))
        labels = torch.load(os.path.join('./dataset/',opt.model, 'labels.pt'))
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs_att = attack_net(inputs)

        _, predicted_att = torch.max(outputs_att.data, 1)
        total += labels.size(0)

        correct_att += (predicted_att.to(device) == labels).sum()
        if target_net is not None:
            outputs_tar = target_net(inputs * max_pixel)
            _, predicted_tar = torch.max(outputs_tar.data, 1)
            correct_tar += (predicted_tar.to(device) == labels).sum()

        print('Accuracy of the reverse engineered model produced by REOM: %.2f %%' %
                (100. * correct_att.float() / total))
        if target_net is not None:
            print('Accuracy of the source TFLite model: %.2f %%' %
                    (100. * correct_tar.float() / total))
        print('total samples: %d' % total)

def test_adver(net, tar_net, attack, dtype='uint8',white_box=False, clip_min=0.0, clip_max=1.0, target=False, nb_batch=10):
    net.eval()
    # tar_net.eval()
    # BIM
    if dtype == 'uint8':
        max_pixel = 255.0
    elif dtype == 'float32':
        max_pixel = 1.0
    if attack == 'BIM':
        fmodel = fb.PyTorchModel(net, bounds=(clip_min,clip_max))
        attack_fb = fb.attacks.L2BasicIterativeAttack(abs_stepsize=opt.eps_iter, steps=opt.nb_iter, random_start=False)
    elif attack == 'PGD':
        fmodel = fb.PyTorchModel(net, bounds=(clip_min,clip_max))
        attack_fb = fb.attacks.L2ProjectedGradientDescentAttack(abs_stepsize=opt.eps_iter, steps=opt.nb_iter, random_start=False)
    elif attack == 'FGSM':
        fmodel = fb.PyTorchModel(net, bounds=(clip_min,clip_max))
        attack_fb = fb.attacks.L2FastGradientAttack(random_start=False)
    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0
    att_num = 0.
    acc_num = 0.

    inputs = torch.load(os.path.join('./dataset/',opt.model, 'inputs.pt'))
    labels = torch.load(os.path.join('./dataset/',opt.model, 'labels.pt'))
    inputs = inputs.to(device)
    labels = labels.to(torch.int64).to(device)
    with torch.no_grad():
        if not white_box:
            outputs = tar_net(inputs * max_pixel)
        else:
            outputs = net(inputs)
    nb_class = outputs.size(1)
    _, predicted = torch.max(outputs.data, 1)
    if target:
        labels = torch.randint(0, nb_class, (inputs.size(0),)).to(torch.int64).to(device)

        ones = torch.ones_like(predicted).to(device)
        zeros = torch.zeros_like(predicted).to(device)
        acc_sign = torch.where(predicted.to(device) == labels, zeros, ones)
        acc_num += acc_sign.sum().float()
        _, adv_inputs_ori, is_adv = attack_fb(fmodel, inputs, TargetedMisclassification(labels), epsilons=opt.eps)
        L2_distance = (adv_inputs_ori - inputs).squeeze()
        L2_distance = (torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1)).data
        L2_distance = L2_distance * acc_sign
        total_L2_distance += L2_distance.sum()
        with torch.no_grad():
            outputs = tar_net(adv_inputs_ori*max_pixel)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels).sum()
            att_sign = torch.where(predicted.to(device) == labels, ones, zeros)
            att_sign = att_sign + acc_sign
            att_sign = torch.where(att_sign == 2, ones, zeros)
            att_num += att_sign.sum().float()


    else:
        ones = torch.ones_like(predicted).to(device)
        zeros = torch.zeros_like(predicted).to(device)
        acc_sign = torch.where(predicted.to(device) == labels, ones, zeros)
        acc_num += acc_sign.sum().float()

        _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, Misclassification(labels), epsilons=opt.eps)
        L2_distance = (adv_inputs_ori.to(device) - inputs).squeeze()
        L2_distance = (torch.linalg.norm(torch.flatten(L2_distance, start_dim=1), dim=1)).data
        L2_distance = L2_distance * acc_sign
        total_L2_distance += L2_distance.sum()
        with torch.no_grad():
            outputs = tar_net(adv_inputs_ori*max_pixel)
            # outputs = attack_net(adv_inputs_ori)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels).sum()
            att_sign = torch.where(predicted.to(device) == labels, zeros, ones)
            att_sign = att_sign + acc_sign
            att_sign = torch.where(att_sign == 2, ones, zeros)
            att_num += att_sign.sum().float()

    if target:
        print('Targeted attack success rate: %.2f %%' %
              ((att_num / acc_num * 100.0)))
    else:
        print('Non-targeted attack success rate: %.2f %%' %
              (att_num / acc_num * 100.0))
    print('l2 distance:  %.4f ' % (total_L2_distance / acc_num))


if opt.model == 'bird':

    dtype='uint8'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/bird.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/bird.pth').to(device).eval()

elif opt.model == 'insect':

    dtype='uint8'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/insect.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/insect.pth').to(device).eval()

elif opt.model == 'plant':

    dtype='uint8'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/plant.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/plant.pth').to(device).eval()

elif opt.model == 'plant_disease':

    dtype='float32'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/plant_disease.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/plant_disease.pth').to(device).eval()

elif opt.model == 'american_sign_language':

    dtype='float32'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/american_sign_language.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/american_sign_language.pth').to(device).eval()

elif opt.model == 'cassava':

    dtype='uint8'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)

    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/cassava.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/cassava.pth').to(device).eval()

elif opt.model == 'fruit':

    dtype = 'float32'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=-1.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/fruit.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/fruit.pth').to(device).eval()

elif opt.model == 'skin':

    dtype = 'float32'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=-1.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)
    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/skin.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/skin.pth').to(device).eval()

elif opt.model == 'imagenet':
    dtype='uint8'
    kwargs = dict(dtype=dtype)
    kwargs.update(clip_min=0.0)
    kwargs.update(clip_max=1.0)
    kwargs.update(white_box=opt.white_box)
    kwargs.update(target=opt.target)

    tflite_model = tf.lite.Interpreter(model_path='./tflite_model/imagenet.tflite', num_threads=num_threads)                                        
    target_net = tf_inference(tflite_model, expand=0, dtype=dtype)           
    attack_net = torch.load('./pytorch_model/imagenet.pth').to(device).eval()

test_acc(attack_net, target_net=target_net, dtype=dtype)
print('Attacking algorithm: %s' % opt.adv)
test_adver(attack_net, target_net, opt.adv, **kwargs)
print('------------------------------------------------------')