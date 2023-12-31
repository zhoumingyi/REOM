import os
import argparse
import torch
import numpy as np
import onnx
import onnxruntime as ort
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from onnx_pytorch.onnx_pytorch import ConvertModel
from modifier.pruning import pruning
from modifier.translation import translation
from modifier.auto_matching import auto_matching
from utils.utils import *

SEED = 100
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='path of tflite model', default='imagenet')
parser.add_argument('--inner_out', type=str, help='node name', default=None)
parser.add_argument('--alpha', type=int, default=0.0, help='threshold for auto-matching')
parser.add_argument('--all', action='store_true')
parser.add_argument('--save_onnx', action='store_true')
parser.add_argument('--acc_mode', action='store_true')       # When enable the acc_mode, some rule of pruning module will be skipped.
opt = parser.parse_args()
print(opt)

def onnx_modifier(onnx_model, acc_mode=False, tflite_model=None):
    pruning(onnx_model, acc_mode=acc_mode)
    translation(onnx_model)
    auto_matching(onnx_model, similarity=opt.alpha, tflite_model=tflite_model)

if not opt.all:
    tflite_model_path = './tflite_model/'
    model_path = tflite_model_path + opt.model_name + '.tflite'
    inputs = generate_random_data(model_path)
    tflite_out, output_details = test_tflite_results(model_path, inputs, output_id=0, inter_out=opt.inner_out)

    if opt.save_onnx == True:
        TfliteToOnnx(tflite_model_path, opt.model_name + '.tflite')
    onnx_model = onnx.load('./out_model/'+opt.model_name+'.onnx')
    onnx_modifier(onnx_model, acc_mode=opt.acc_mode, tflite_model=model_path)
    onnx.save(onnx_model, './out_model/'+opt.model_name+'_modified.onnx')
    # onnx.checker.check_model(onnx_model)
    pytorch_model = ConvertModel(onnx_model, experimental=True).eval()
    torch.save(pytorch_model, './pytorch_model/'+opt.model_name+'.pth')

else:
    tflite_path = './tflite_model/'
    filelist = os.listdir(tflite_path)
    # print(filelist)
    for i in range(len(filelist)):    # 153, 214
        if not filelist[i].endswith('.tflite'):
            continue
        print(i, "_", "model name:", filelist[i])
        model_path = tflite_path+filelist[i]
        interpreter = tf.lite.Interpreter(model_path=model_path)
        if opt.save_onnx == True:
            TfliteToOnnx(tflite_path, filelist[i])
        # inputs = generate_random_data(model_path)
        input_details = interpreter.get_input_details()
        inputs = []
        if input_details[0]['dtype'] == numpy.uint8:
            inputs.append(numpy.expand_dims(((torch.load(os.path.join('./dataset/',os.path.splitext(filelist[i])[0], 'inputs.pt')).permute(0,2,3,1)) * 255.0).to(torch.uint8).numpy()[0], axis=0))
            data_scale = 255.0
        elif input_details[0]['dtype'] == numpy.float32:
            inputs.append(numpy.expand_dims(torch.load(os.path.join('./dataset/',os.path.splitext(filelist[i])[0], 'inputs.pt')).permute(0,2,3,1).numpy()[0], axis=0))
            data_scale = 1.0
        print(inputs[0].max(), inputs[0].min())
        tflite_out, output_details = test_tflite_results(model_path, inputs, output_id=0, inter_out=opt.inner_out)
        _, predicted_att = torch.max(torch.from_numpy(tflite_out), 1)
        print("tflite_out:", predicted_att)
        onnx_model = onnx.load('./out_model/'+os.path.splitext(filelist[i])[0]+'.onnx')
        onnx_modifier(onnx_model, acc_mode=opt.acc_mode, tflite_model=model_path)
        onnx.save(onnx_model, './out_model/'+os.path.splitext(filelist[i])[0]+'_modified.onnx')
        pytorch_model = ConvertModel(onnx_model).eval()
        output_pt = pytorch_model(torch.from_numpy(inputs[0]).to(torch.float32)/data_scale)
        _, predicted_att = torch.max(output_pt.data, 1)
        print('Pytorch output:', predicted_att)
        if torch.is_tensor(output_pt):
            error = np.absolute(output_pt.detach().squeeze().to(torch.float).numpy()-tflite_out.squeeze())
        else:
            error = np.absolute(output_pt[output_details[0]['name']].detach().squeeze().to(torch.float).numpy()-tflite_out.squeeze())
        if tflite_out.dtype == 'uint8':
            print("data_type is uint8")
            scale = 256.0
        else:
            scale = (tflite_out.max() - tflite_out.min() + 1.0e-08)
        print('final_men_error_normlized:', error.mean() / scale)
        print('final_max_error_normlized:', error.max() / scale)
