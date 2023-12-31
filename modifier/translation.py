from copy import deepcopy
import onnx
from utils.utils import *


def translation(onnx_model):
    model_copy = deepcopy(onnx_model)
    translated_layers = []
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
    layer_count = 0
    for node_trans in model_copy.graph.node:
        layer_count += 1
        if node_trans.op_type == 'GlobalAveragePool':
            for node in model_copy.graph.node:
                if node.op_type == 'Squeeze' and node.input[0] == node_trans.output[0]:
                    # print(list(range(2, len(onnx_model.graph.input[0].type.tensor_type.shape.dim))))
                    mean_node = onnx.helper.make_node("ReduceMean",
                                            name=node_trans.name,
                                            inputs=[node_trans.input[0]],
                                            outputs=[node.output[0]],
                                            axes=list(range(2, len(onnx_model.graph.input[0].type.tensor_type.shape.dim))),
                                            keepdims=0,)
                    onnx_model.graph.node.insert(layer_count, mean_node)    
                    layer_count += 1                
                    translated_layers.append(node_trans)
                    translated_layers.append(node)
        if node_trans.op_type == 'GlobalMaxPool':
            flatten_node = onnx.helper.make_node("Flatten",
                                    name=node_trans.name,
                                    inputs=[node_trans.input[0]],
                                    outputs=[node_trans.output[0]+'_flatten'],
                                    axes=1,)
            onnx_model.graph.node.insert(layer_count, flatten_node)    
            layer_count += 1 
            max_node = onnx.helper.make_node("Max",
                                    name=node_trans.name,
                                    inputs=[node_trans.output[0]+'_flatten'],
                                    outputs=[node_trans.output[0]],)
            onnx_model.graph.node.insert(layer_count, max_node)  
            layer_count += 1                
            translated_layers.append(node_trans)
        # if node_trans.op_type == 'SpaceToDepth':
            # Has been moved to the onnx2pytorch
        # if node_trans.op_type == 'DepthToSpace':
            # Has been moved to the onnx2pytorch
        if node_trans.op_type == 'ConvInteger':
            sub_node = onnx.helper.make_node("Sub",
                                    name=node_trans.name,
                                    inputs=[node_trans.input[0], 'zeropoint_'+node_trans.name],
                                    outputs=[node_trans.input[0]+'sub'],)
            sub_node_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # vals=weights[node_dequant.input[2]].raw_data,
                                        vals=Torch2OnnxWeights(OnnxWeights2Torch(weights[node_trans.input[2]]).to(torch.float)).raw_data,
                                        raw=True
                                        )
            onnx_model.graph.node.insert(layer_count, sub_node)  
            onnx_model.graph.initializer.append(sub_node_tensor)  
            layer_count += 1 
            node_trans.op_type = 'Conv'
            node_trans.inputs = [node_trans.input[0], node_trans.input[1]]
            node_trans.input[0] = node_trans.input[0]+'sub'
        if node_trans.op_type == 'MatMulInteger':
            sub_node = onnx.helper.make_node("Sub",
                                    name=node_trans.name,
                                    inputs=[node_trans.input[0], 'zeropoint_'+node_trans.name],
                                    outputs=[node_trans.input[0]+'sub'],)
            sub_node_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # vals=weights[node_dequant.input[2]].raw_data,
                                        vals=Torch2OnnxWeights(OnnxWeights2Torch(weights[node_trans.input[2]]).to(torch.float)).raw_data,
                                        raw=True
                                        )
            onnx_model.graph.node.insert(layer_count, sub_node)  
            onnx_model.graph.initializer.append(sub_node_tensor)  
            layer_count += 1 
            node_trans.op_type = 'MatMul'
            node_trans.inputs = [node_trans.input[0], node_trans.input[1]]
            node_trans.input[0] = node_trans.input[0]+'sub'


        if node_trans.op_type == 'DequantizeLinear':
            if node_trans.input[0] in weights:
                continue
            else:
                dequant_add = onnx.helper.make_node("Sub",
                                        name='dequantsub_'+node_trans.name,
                                        inputs=[node_trans.input[0], 'zeropoint_'+node_trans.name],
                                        outputs=['sub_result_'+node_trans.name])
                # print(OnnxWeights2Torch(weights[node_dequant.input[2]]).item())
                # todo: check the size not match in make tensor
                dequant_add_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # vals=weights[node_dequant.input[2]].raw_data,
                                        vals=Torch2OnnxWeights(OnnxWeights2Torch(weights[node_trans.input[2]]).to(torch.float)).raw_data,
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_add)
                # print(layer_count)
                # add_layers.append(dequant_add)
                onnx_model.graph.initializer.append(dequant_add_tensor)
                layer_count += 1
                dequant_mul = onnx.helper.make_node("Mul",
                                        name='dequantmul_'+node_trans.name,
                                        inputs=['sub_result_'+node_trans.name, 'scale_'+node_trans.name],
                                        outputs=[node_trans.output[0]])
                dequant_mul_tensor = onnx.helper.make_tensor('scale_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[1]]).size(),
                                        vals=weights[node_trans.input[1]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_dequant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_mul)
                # print(layer_count)
                # add_layers.append(dequant_mul)
                onnx_model.graph.initializer.append(dequant_mul_tensor)
                layer_count += 1
            translated_layers.append(node_trans)
        elif node_trans.op_type == 'QuantizeLinear':
            if node_trans.input[0] in weights:
                continue
            else:
                dequant_div = onnx.helper.make_node("Div",
                                        name='quantdiv_'+node_trans.name,
                                        inputs=[node_trans.input[0], 'scale_'+node_trans.name],
                                        outputs=['div_result_'+node_trans.name])
                dequant_mul_tensor = onnx.helper.make_tensor('scale_'+node_trans.name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[1]]).size(),
                                        vals=weights[node_trans.input[1]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_quant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_div)
                # add_layers.append(dequant_div)
                onnx_model.graph.initializer.append(dequant_mul_tensor)
                layer_count += 1
                dequant_add = onnx.helper.make_node("Add",
                                        name='quantadd_'+node_trans.name,
                                        inputs=['div_result_'+node_trans.name, 'zeropoint_'+node_trans.name],
                                        outputs=[node_trans.output[0]])

                dequant_add_tensor = onnx.helper.make_tensor('zeropoint_'+node_trans.name,
                                        data_type=onnx.TensorProto.UINT8,
                                        dims=OnnxWeights2Torch(weights[node_trans.input[2]]).size(),
                                        # dims=[1],
                                        vals=weights[node_trans.input[2]].raw_data,
                                        # vals=[OnnxWeights2Torch(weights[node_quant.input[2]]).item()],
                                        raw=True
                                        )
                onnx_model.graph.node.insert(layer_count, dequant_add)
                # add_layers.append(dequant_add)
                onnx_model.graph.initializer.append(dequant_add_tensor)
                layer_count += 1
            translated_layers.append(node_trans)
    # for node_trans in onnx_model.graph.node:
        elif node_trans.op_type == 'Sum':
            node_trans.op_type = 'Add'
        elif node_trans.op_type == 'LRN':
            node_trans.op_type = 'LocalResponseNorm'
    for layer in translated_layers:
        # print(layer.op_type)
        onnx_model.graph.node.remove(layer)