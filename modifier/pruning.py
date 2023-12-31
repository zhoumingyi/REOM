from copy import deepcopy
import onnx
from utils.utils import *


def pruning(onnx_model, acc_mode):
    model_copy = deepcopy(onnx_model)
    redundant_layers = []
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    redundant_sign = False
    layer_count = 0
    for node_pruning in model_copy.graph.node:
        # print(node_pruning.name)
        layer_count += 1
        if node_pruning.op_type == 'DequantizeLinear':
            if node_pruning.input[0] in weights:
                # layer_count += 1
                if len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 4:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 5:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 3:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 2:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1).to(torch.float)
                elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 1:
                    dequant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]]).to(torch.float)\
                        - OnnxWeights2Torch(weights[node_pruning.input[2]]).to(torch.float))\
                        * OnnxWeights2Torch(weights[node_pruning.input[1]]).to(torch.float)

                for node in onnx_model.graph.node:
                    for i in range(len(node.input)):
                        if node.input[i] == node_pruning.output[0]:
                            # print('Find a quantized param:', node_input)
                            # print('intput name:', node.input[i])
                            node.input[i] = node.input[i] + "_modified"
                            dequant_tensor = onnx.helper.make_tensor(node.input[i],
                                                                    data_type=onnx.TensorProto.FLOAT,
                                                                    dims=dequant_weights.size(),
                                                                    vals=Torch2OnnxWeights(dequant_weights).raw_data,
                                                                    raw=True)
                            onnx_model.graph.initializer.append(dequant_tensor)
                # print(node_dequant.name, dequant_weights)
                redundant_layers.append(node_pruning)
        if node_pruning.op_type == 'DynamicQuantizeLinear':
            for node in onnx_model.graph.node:
                if node.op_type == 'DequantizeLinear' and node.input[0] == node_pruning.output[0]\
                     and node.input[1] == node_pruning.output[1]\
                     and node.input[2] == node_pruning.output[2]:
                    try:
                        weights[node_pruning.input[0]]
                    except:
                        for node_restore in onnx_model.graph.node:
                            for i in range(len(node_restore.input)):
                                if node.output[0] == node_restore.input[i]:
                                    node_restore.input[i] = node_pruning.input[0]
                    else:
                        for node_restore in onnx_model.graph.node:
                            for i in range(len(node_restore.input)):
                                if node.output[0] == node_restore.input[i]:
                                    restore_tensor = onnx.helper.make_tensor(node_restore.input[i],
                                                                            data_type=onnx.TensorProto.FLOAT,
                                                                            dims=OnnxWeights2Torch(weights[node_pruning.input[0]]).size(),
                                                                            # dims=[128],
                                                                            vals=weights[node_pruning.input[0]].raw_data,
                                                                            raw=True)
                                    onnx_model.graph.initializer.append(restore_tensor)
                                # else:
                                #     raise Exception("Ohhhhh Mingyi, type error for remove_DynamicQuant_layers!")
                    redundant_layers.append(node_pruning)
                    redundant_layers.append(node)
        if node_pruning.op_type == 'QuantizeLinear':
            has_quant_dequant_comb = False
            for node_dequant in onnx_model.graph.node: 
                if node_dequant.op_type == 'DequantizeLinear' and (node_pruning.output[0] in node_dequant.input):
                    if weights[node_pruning.input[1]].raw_data == weights[node_dequant.input[1]].raw_data and \
                        weights[node_pruning.input[2]].raw_data == weights[node_dequant.input[2]].raw_data and \
                        0.02352 <= OnnxWeights2Torch(weights[node_pruning.input[1]]) <= 0.02353:
                        # print("find a combo")
                        try:
                            weights[node_pruning.input[0]]
                        except:
                            for node_restore in onnx_model.graph.node:
                                for i in range(len(node_restore.input)):
                                    if node_dequant.output[0] == node_restore.input[i]:
                                        # node_restore.input[i] = node_pruning.input[0] 
                                        quant_relu = onnx.helper.make_node("Relu",
                                                    name='relu6_'+node_pruning.name,
                                                    inputs=[node_pruning.input[0]],
                                                    outputs=[node_restore.input[i]])
                                        # print(layer_count)
                            onnx_model.graph.node.insert(layer_count+1, quant_relu)
                            layer_count += 1
                                        # continue
                        else:
                            for node_restore in onnx_model.graph.node:
                                for i in range(len(node_restore.input)):
                                    if node_dequant.output[0] == node_restore.input[i]:
                                        restore_tensor = onnx.helper.make_tensor(node_restore.input[i],
                                                        data_type=onnx.TensorProto.FLOAT,
                                                        dims=OnnxWeights2Torch(weights[node_pruning.input[0]]).size(),
                                                        # dims=[128],
                                                        vals=weights[node_pruning.input[0]].raw_data,
                                                        raw=True)
                                        onnx_model.graph.initializer.append(restore_tensor)
                        redundant_layers.append(node_pruning)
                        redundant_layers.append(node_dequant)

                        has_quant_dequant_comb = True
                # elif node_dequant.op_type == 'DequantizeLinear' and \
                #     weights[node_pruning.input[1]].raw_data == weights[node_dequant.input[1]].raw_data and \
                #     weights[node_pruning.input[2]].raw_data == weights[node_dequant.input[2]].raw_data and \
                #     (not (0.02352 <= OnnxWeights2Torch(weights[node_pruning.input[1]]) <= 0.02353)):
                #     for node_next in onnx_model.graph.node:
                #         for i in range(len(node_next.input)):
                #             if node_pruning.output[0] == node_next.input[i]:
                #                 node_next.input[i] = node_pruning.input[0]
                #             elif node_dequant.output[0] == node_next.input[i]: 
                #                 node_next.input[i] = node_dequant.input[0]
                #     redundant_layers.append(node_pruning)
                #     redundant_layers.append(node_dequant)

                #     has_quant_dequant_comb = True
            if not has_quant_dequant_comb:
                if node_pruning.input[0] in weights:
                    if len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 4:
                        quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                            / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1)\
                            + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)
            
                    elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 5:
                        quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                            / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1, 1, 1)\
                            + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                    elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 3:
                        quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                            / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1, 1)\
                            + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                    elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 2:
                        quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                            / OnnxWeights2Torch(weights[node_pruning.input[1]]).reshape(-1, 1)\
                            + OnnxWeights2Torch(weights[node_pruning.input[2]]).reshape(-1, 1).to(torch.float)).clip(0,255).to(torch.uint8)

                    elif len(OnnxWeights2Torch(weights[node_pruning.input[0]]).size()) == 1:
                        quant_weights = (OnnxWeights2Torch(weights[node_pruning.input[0]])\
                            / OnnxWeights2Torch(weights[node_pruning.input[1]])\
                            + OnnxWeights2Torch(weights[node_pruning.input[2]]).to(torch.float)).clip(0,255).to(torch.uint8)

                    for node in onnx_model.graph.node:
                        for i in range(len(node.input)):
                            if node.input[i] == node_pruning.output[0]:
                                # print('Find a quantized param:', node_input)
                                node.input[i] = node.input[i] + "_modified"
                                quant_tensor = onnx.helper.make_tensor(node.input[i],
                                                                        data_type=onnx.TensorProto.UINT8,
                                                                        dims=quant_weights.size(),
                                                                        vals=Torch2OnnxWeights(quant_weights).raw_data,
                                                                        # vals=quant_weights.numpy(),
                                                                        raw=True
                                                                        )
                                onnx_model.graph.initializer.append(quant_tensor)
                    redundant_layers.append(node_pruning)
        if node_pruning.op_type == 'Gemm':
            params = [weights[par_name] for par_name in node_pruning.input if par_name in weights]
            if len(params) == 3:
                kwargs = extract_attributes(node_pruning)
                # print(kwargs)
                if not kwargs["transpose_activation"]:
                    A = OnnxWeights2Torch(weights[node_pruning.input[0]])
                else:
                    A = OnnxWeights2Torch(weights[node_pruning.input[0]]).t()
                if not kwargs["transpose_weight"]:
                    B = OnnxWeights2Torch(weights[node_pruning.input[1]])
                else:
                    B = OnnxWeights2Torch(weights[node_pruning.input[1]]).t()
                C = OnnxWeights2Torch(weights[node_pruning.input[2]])
                actual_tensor = torch.nn.functional.linear(input=A, weight=B, bias=C)
                Gemm_add_tensor = onnx.helper.make_tensor(node_pruning.output[0],
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=actual_tensor.size(),
                                            # dims=[1],
                                            # vals=weights[node_quant.input[2]].raw_data,
                                            vals=Torch2OnnxWeights(actual_tensor).raw_data,
                                            raw=True
                                            )

                redundant_layers.append(node_pruning)
                onnx_model.graph.initializer.append(Gemm_add_tensor)
        if node_pruning.op_type == 'Reshape':
            params = [weights[par_name] for par_name in node_pruning.input if par_name in weights]
            if len(params) == 2:
                print("in the reshape")
                # print(tuple(OnnxWeights2Torch(weights[node_pruning.input[1]]).tolist()))
                reshaped_tensor = torch.reshape(OnnxWeights2Torch(weights[node_pruning.input[0]]), tuple(OnnxWeights2Torch(weights[node_pruning.input[1]]).tolist()))
                reshaped_tensor_para = onnx.helper.make_tensor(node_pruning.output[0],
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=reshaped_tensor.size(),
                                            # dims=[1],
                                            # vals=weights[node_quant.input[2]].raw_data,
                                            vals=Torch2OnnxWeights(reshaped_tensor).raw_data,
                                            raw=True
                                            )
                redundant_layers.append(node_pruning)
                onnx_model.graph.initializer.append(reshaped_tensor_para)
        if node_pruning.op_type == 'Unsqueeze':
            if node_pruning.input[0] in weights:
                kwargs = extract_attributes(node_pruning)
                axis = kwargs['dim']
                unsqueezed_tensor = torch.from_numpy(np.expand_dims(OnnxWeights2Numpy(weights[node_pruning.input[0]]), axis))
                # unsqueezed_tensor = torch.unsqueeze(OnnxWeights2Torch(weights[node_pruning.input[0]]), dim=OnnxWeights2Torch(weights[node_pruning.input[1]]))
                unsqueezed_tensor_para = onnx.helper.make_tensor(node_pruning.output[0],
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=unsqueezed_tensor.size(),
                                            vals=Torch2OnnxWeights(unsqueezed_tensor).raw_data,
                                            raw=True
                                            )
                redundant_layers.append(node_pruning)
                onnx_model.graph.initializer.append(unsqueezed_tensor_para)

        if node_pruning.op_type == 'Squeeze':
            if node_pruning.input[0] in weights:
                if len(node_pruning.input) == 2:
                    squeezed_tensor = torch.squeeze(OnnxWeights2Torch(weights[node_pruning.input[0]]))
                    squeezed_tensor_para = onnx.helper.make_tensor(node_pruning.output[0],
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=squeezed_tensor.size(),
                                                vals=Torch2OnnxWeights(squeezed_tensor).raw_data,
                                                raw=True
                                                )
                    redundant_layers.append(node_pruning)
                    onnx_model.graph.initializer.append(squeezed_tensor_para)
                elif len(node_pruning.input) == 1:
                    squeezed_tensor = torch.squeeze(OnnxWeights2Torch(weights[node_pruning.input[0]]))
                    squeezed_tensor_para = onnx.helper.make_tensor(node_pruning.output[0],
                                                data_type=onnx.TensorProto.FLOAT,
                                                dims=squeezed_tensor.size(),
                                                vals=Torch2OnnxWeights(squeezed_tensor).raw_data,
                                                raw=True
                                                )
                    redundant_layers.append(node_pruning)
                    onnx_model.graph.initializer.append(squeezed_tensor_para)
        if node_pruning.op_type == 'BitShift':
            if node_pruning.input[0] in weights:
                kwargs = extract_attributes(node_pruning)
                if kwargs["direction"] == "LEFT":
                    shifted_tensor = torch.bitwise_left_shift(OnnxWeights2Torch(weights[node_pruning.input[0]]), OnnxWeights2Torch(weights[node_pruning.input[1]]))
                else:
                    shifted_tensor = torch.bitwise_right_shift(OnnxWeights2Torch(weights[node_pruning.input[0]]), OnnxWeights2Torch(weights[node_pruning.input[1]]))
                shifted_tensor_para = onnx.helper.make_tensor(node_pruning.output[0],
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=shifted_tensor.size(),
                                            vals=Torch2OnnxWeights(shifted_tensor).raw_data,
                                            raw=True
                                            )
                redundant_layers.append(node_pruning)
                onnx_model.graph.initializer.append(shifted_tensor_para)                

        if redundant_sign:
            redundant_layers.append(node_pruning)
        if node_pruning.op_type == 'Softmax' and acc_mode==False:
            redundant_sign = True
            onnx_model.graph.output[0].name = node_pruning.output[0]
    for layer in redundant_layers:
        # print(layer.name)
        onnx_model.graph.node.remove(layer)

    redundant_layers = []
    layer_count = 0
    for node_pruning in onnx_model.graph.node:
        # print(node_pruning.name)
        layer_count += 1
        if node_pruning.op_type == 'Transpose' and node_pruning.attribute[0].ints == [0,3,1,2] and layer_count < 2 and acc_mode==False:
            # print("removed transpose: ", layer_count)
            redundant_layers.append(node_pruning)
            for node_next in onnx_model.graph.node:
                for i in range(len(node_next.input)):
                    if node_pruning.output[0] == node_next.input[i]:
                        node_next.input[i] = node_pruning.input[0]
                        # onnx_model.graph.node.remove(node_pruning)
        if node_pruning.op_type == 'DequantizeLinear' and layer_count < 3 and \
            OnnxWeights2Torch(weights[node_pruning.input[2]]).to(torch.float) == 128.0:
            # print("fake quant")
            # print(layer_count)
            if 0.0078740 <= OnnxWeights2Torch(weights[node_pruning.input[1]]) <= 0.0078741:
            # if 0.0078 <= OnnxWeights2Torch(weights[node_pruning.input[1]]) <= 0.0079:
                for node_next in onnx_model.graph.node:
                    for i in range(len(node_next.input)):
                        if node_pruning.output[0] == node_next.input[i]:
                            node_next.input[i] = node_pruning.input[0]
                redundant_layers.append(node_pruning)
                # onnx_model.graph.node.remove(node_pruning)
            elif OnnxWeights2Torch(weights[node_pruning.input[1]]) == 0.0078125:
                print('input range: 256')
                float2uint8_mul = onnx.helper.make_node("Mul",
                                    name='float2uint8',
                                    inputs=['float_'+onnx_model.graph.input[0].name, 'float2uint8_scale'],
                                    outputs=[onnx_model.graph.input[0].name])   
                float2uint8_mul_tensor = onnx.helper.make_tensor('float2uint8_scale',
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[1],
                                        vals=[256.0]
                                        )
                onnx_model.graph.node.insert(0, float2uint8_mul)
                onnx_model.graph.initializer.append(float2uint8_mul_tensor)
                onnx_model.graph.input[0].name = 'float_'+onnx_model.graph.input[0].name
                layer_count += 1
            elif onnx_model.graph.input[0].type.tensor_type.elem_type == 2:
                # print('input range: 255')
                float2uint8_mul = onnx.helper.make_node("Mul",
                                    name='float2uint8',
                                    inputs=['float_'+onnx_model.graph.input[0].name, 'float2uint8_scale'],
                                    outputs=[onnx_model.graph.input[0].name])   
                float2uint8_mul_tensor = onnx.helper.make_tensor('float2uint8_scale',
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[1],
                                        vals=[255.0]
                                        )
                onnx_model.graph.node.insert(layer_count, float2uint8_mul)
                onnx_model.graph.initializer.append(float2uint8_mul_tensor)
                onnx_model.graph.input[0].name = 'float_'+onnx_model.graph.input[0].name
                layer_count += 1
    for layer in redundant_layers:
    #     # print(layer.name)
        onnx_model.graph.node.remove(layer)