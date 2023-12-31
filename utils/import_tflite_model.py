import tensorflow as tf
import numpy as np
import torch
import time

# input_data_tf = np.array(np.random.random_sample([32, 1, 224, 224, 3]), dtype=np.float32)
# input_data = torch.rand(32, 1, 224, 224, 3)
class tf_inference():
    def __init__(self, model, expand=99, dtype='uint8'):
        # self.input = input
        self.model = model
        self.expand = expand
        self.dtype = dtype
        # interpreter = self.model
        # input_details = self.model.get_input_details()
        # output_details = self.model.get_output_details()

    def __call__(self, inputs):
        inputs = torch.permute(inputs, (0,2,3,1)).cpu().numpy()
        # print(inputs.shape)
        results = torch.from_numpy(self.inference(inputs, expand=self.expand, dtype=self.dtype))
        # print(results.size())
        return results

    def inference(self, inputs, expand, dtype):
        dim = inputs.shape[0]
        # results = np.array(np.random.random_sample([dim, class_num]), dtype=np.uint8)
        results = []
        for i in range(dim):
            results.append(self.andoid_model(inputs[i], expand=expand, dtype=dtype))
        # results = self.andoid_model(inputs, expand=expand, dtype=dtype)
        results = np.array(results).squeeze()
        # return list(results)
        return results

    def andoid_model(self, input, expand, dtype):

        # Load TFLite model and allocate tensors
        # interpreter = tf.lite.Interpreter(model_path="/datasata/mingyi/ondevice/DL_models/fine_tuned/mobilenet.letgo.v1_1.0_224_quant.v7.tflite")
        interpreter = self.model

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        # print(str(input_details))
        output_details = interpreter.get_output_details()
        # print(str(output_details))
        # Test model on input data
        if expand == 99:
            # input_data = input.squeeze()
            input_data = input
        else:
            input_data = np.expand_dims(input.squeeze(), axis=expand)
        if dtype == 'uint8':
            input_data = input_data.astype(np.uint8)
        elif dtype == 'int32':
            input_data = input_data.astype(np.int32)
        elif dtype == 'float32':
            input_data = input_data.astype(np.float32)
        # print(input_data.shape)
        # print(input_data.max(), input_data.min())
        # interpreter.set_tensor(input_details[0]['index'], input_data)
        # interpreter.resize_tensor_input(input_details[0]['index'],[input_data.shape[0], input_data.shape[1], input_data.shape[2], input_data.shape[3]])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])
        # result = np.squeeze(output_data)
        # result = (output_data / 255.0)
        # if type(result) == 'uint8':
        #     result = result.astype(np.float32)
        #     result = (result / 255.0)
        # print(result.sum())
        return result
