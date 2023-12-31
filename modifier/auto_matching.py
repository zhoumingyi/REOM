from copy import deepcopy
import difflib
import onnx
# from tokenize import String
import onnxruntime as ort
from utils.utils import *

'''This is the corresponding supported list for ONNX operators'''
supported_list = [            
    "Add","AveragePool","BatchNormalization","Cast","Ceil","Clip","Concat","Constant","ConstantOfShape","Conv","ConvTranspose","Div","Elu","Equal","Erf","Exp","Expand","Flatten","Floor","Gather","GatherND","Gemm","GlobalAveragePool","Greater","Identity","InstanceNormalization","LeakyRelu","Less","Log","Loop","LSTM","MatMul","Max","MaxPool","Min","Mul","NonMaxSuppression","Not","OneHot","Or","Pad","Pow","PRelu","Range","Reciprocal","ReduceMean","ReduceProd","ReduceSum","Relu","Reshape","Resize","Scatter","ScatterElements","ScatterND","Shape","Sigmoid","Slice","Softmax","Softplus","Softsign","Split","Sqrt","Squeeze","Sub","Tanh","ThresholdedRelu","Tile","TopK","Transpose","Unsqueeze","Upsample","Where","LpNormalization","SpaceToDepth","DepthToSpace","HardSwish","Einsum","Roll","GreaterOrEqual","GlobalMaxPool","LessOrEqual","ResizeNearestNeighbor","ReduceMax","ReduceMin","Dropout","LocalResponseNorm","ArgMax","Neg"
]

def out_similarity(tflite_model, onnx_model):
    inputs = generate_random_data(tflite_model)
    tflite_out, output_details = test_tflite_results(tflite_model, inputs, output_id=0)
    ort_sess = ort.InferenceSession(onnx_model)
    outputs = ort_sess.run(None, {'normalized_input_image_tensor': inputs[0]})
    print((tflite_out - outputs[0]).mean / (tflite_out.max() - tflite_out.min()))
    
def operation_similarity(l1: str, l2: str):
    return difflib.SequenceMatcher(None, str.lower(l1), str.lower(l2)).quick_ratio()

def auto_matching(onnx_model, similarity=0.0, tflite_model=None):
    model_copy = deepcopy(onnx_model)
    nonsupported_layer = []
    layer_count = 0
    for node_nonsupported in onnx_model.graph.node:
        layer_count += 1
        if not (node_nonsupported.op_type in supported_list):
            print("matching op:", node_nonsupported.op_type)
            most_similar_op = None
            # similarity = 0.
            most_similarity = 0.
            for i in range(len(supported_list)):
                if operation_similarity(node_nonsupported.op_type, supported_list[i]) > similarity and operation_similarity(node_nonsupported.op_type, supported_list[i]) > most_similarity:
                    most_similarity = operation_similarity(node_nonsupported.op_type, supported_list[i])
                    most_similar_op = supported_list[i]
            # print('matched operator:', most_similar_op)
            # supported_op = onnx.helper.make_node(most_similar_op,
            #                         name=node_nonsupported.name + '_matched',
            #                         inputs=node_nonsupported.input,
            #                         outputs=node_nonsupported.output
            #                         )
            # nonsupported_layer.append(node_nonsupported)

            # onnx_model.graph.node.insert(layer_count, supported_op)
            # layer_count += 1
            pre_type = node_nonsupported.op_type
            node_nonsupported.op_type = most_similar_op
            # if out_similarity(tflite_model, onnx_model) > 0.1:
            #     node_nonsupported.op_type = pre_type

    # for layer in nonsupported_layer:
    #     # print(layer.name)
    #     onnx_model.graph.node.remove(layer)

