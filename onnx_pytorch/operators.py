from collections import defaultdict
from functools import partial

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from onnx import numpy_helper
from torch.nn.modules.linear import Identity

from .attribute import extract_attributes
# from onnx2pytorch.convert.layer import (
#     convert_layer,
#     convert_linear_layer,
#     convert_batch_norm_layer,
#     convert_instance_norm_layer,
#     convert_lstm_layer,
# )
from .layer import (
    convert_layer,
    convert_linear_layer,
    convert_batch_norm_layer,
    convert_instance_norm_layer,
    convert_lstm_layer,
)
from onnx2pytorch.operations import *
from .pad import Pad
from .resize import Resize
from onnx2pytorch.operations.base import OperatorWrapper
from onnx2pytorch.operations import Upsample
from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
    value_wrapper,
)
from .reshape import Reshape
# ---------------------------
# my layer definations
from onnx2pytorch.operations.base import Operator
class Normalization(nn.Module):
    def __init__(self, dim=-1, p=2):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, inputs: torch.Tensor):
        # print(inputs.size())
        outputs = nn.functional.normalize(inputs, p=self.p, dim=self.dim)
        # print(outputs.size())
        return outputs

class SpaceToDepth(Operator):
    def __init__(self, blocksize=2):
        super().__init__()
        self.blocksize = blocksize

    def forward(self, inputs: torch.Tensor):
        shape = inputs.size()
        data = Reshape(inputs, [shape[0], shape[1], shape[2] // self.blocksize, self.blocksize,
                                 shape[3] // self.blocksize, self.blocksize])
        data = Transpose(data, [0, 3, 5, 1, 2, 4])
        outputs = Reshape(data, [shape[0], shape[1] * (self.blocksize ** 2),
                              shape[2] // self.blocksize, shape[3] // self.blocksize])

        return outputs


class DepthToSpace(Operator):
    def __init__(self, blocksize=2, mode="DCR"):
        super().__init__()
        self.blocksize = blocksize
        self.mode=mode

    def forward(self, inputs: torch.Tensor):
        shape = inputs.size()
        if self.mode == "DCR":
            data = Reshape(inputs, [shape[0], self.blocksize, self.blocksize, shape[1] // (self.blocksize**2), shape[2],
                                    shape[3]])
            data = Transpose(data, [0, 3, 4, 1, 5, 2])
            outputs = Reshape(data, [shape[0], shape[1] // (self.blocksize ** 2),
                                shape[2] * self.blocksize, shape[3] * self.blocksize])
        else:
            data = Reshape(inputs, [shape[0], shape[1] // (self.blocksize**2), self.blocksize, self.blocksize,
                                    shape[2], shape[3]])
            data = Transpose(data, [0, 1, 4, 2, 5, 3])
            data = Reshape(inputs, [shape[0], shape[1] // (self.blocksize ** 2),
                                shape[2] * self.blocksize, shape[3] * self.blocksize])

        return outputs


class HardSwish(nn.Module):

    def forward(self, x: torch.Tensor):
        outputs = x * torch.maximum(torch.zeros_like(x), torch.minimum(torch.ones_like(x), 1/6 * x + 0.5))
        return outputs

# class View(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         shape[0] = -1
#         self.shape = tuple(shape)
#         print(self.shape)

#     def forward(self, x: torch.Tensor):
#         outputs = torch.reshape(x, tuple(self.shape))
#         return outputs


class Roll(nn.Module):
    def __init__(self, shift, axis):
        super().__init__()
        self.shift = shift
        self.dims = axis

    def forward(self, inputs: torch.Tensor):
        outputs = torch.roll(inputs, shifts=self.shift, dims=self.dims)
        return outputs

class EinSum(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outputs = torch.einsum(self.equation, x, y)
        return outputs

class Great_Equal(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outputs = torch.ge(x, y)
        return outputs

class LessOrEqual(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outputs = torch.le(x, y)
        return outputs

class GlobalMaxPool(nn.Module):

    def forward(self, x: torch.Tensor):
        outputs = torch.max(torch.flatten(x, 1), 1)
        return outputs

class ResizeNearestNeighbor(nn.Module):
    def __init__(self, size, align_corners=False, half_pixel_centers=True):
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.half_pixel_centers = half_pixel_centers

    def forward(self, x: torch.Tensor):
        outputs = F.interpolate(x, self.size, align_corners=self.align_corners, mode='nearest')
        return outputs

class ReduceMax(nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        # self.half_pixel_centers = half_pixel_centers

    def forward(self, x: torch.Tensor):
        outputs = torch.max(x, dim=self.dim, keepdim=self.keepdim).values
        return outputs

class ReduceMin(nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        # self.half_pixel_centers = half_pixel_centers

    def forward(self, x: torch.Tensor):
        outputs = torch.min(x, dim=self.dim, keepdim=self.keepdim).values
        return outputs

class Argmax(nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        # self.half_pixel_centers = half_pixel_centers

    def forward(self, x: torch.Tensor):
        outputs = torch.argmax(x, dim=self.dim, keepdim=self.keepdim)
        return outputs

class LocalResponseNorm(nn.Module):
    def __init__(self, alpha, beta, bias, size):
        super().__init__()
        self.lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=bias)

    def forward(self, x: torch.Tensor):
        outputs = self.lrn(x)
        return outputs

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace).eval()

    def forward(self, x: torch.Tensor):
        outputs = self.dropout(x)
        # outputs = x
        return outputs

class quan_add(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if y.dtype == torch.uint8:
            # outputs = torch.round(torch.add(x, y.to(torch.float)).clip(0, 255))
            outputs = torch.add(x, y.to(torch.float)).clip(0, 255)
            # outputs = torch.add(x, y.to(torch.float))
        else:
            raise ValueError(
                "data type is not correct for dequant add."
            )
        return outputs

class test_add(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # outputs = torch.add(x.to(torch.float), y.to(torch.float))
        outputs = x.to(torch.float) + y.to(torch.float)
        return outputs
    

# ---------------------------


def get_buffer_name(param_name):
    """
    Convert name of initializer to valid name for nn.Module attribute.
    """
    return "_initializer_{}".format(param_name.replace(".", "_"))


def get_init_parameter(modules, item, default):
    """
    Look in modules for the item, and if not found return default.

    Parameters
    ----------
    modules: list of nn.Modules
        Modules whose attributes to search.
    item: str
        Name of initializer in ONNX model.
    default: torch.Tensor
        Tensor to return if item not found.
    """
    item_name = get_buffer_name(item)
    for mod in modules:
        if hasattr(mod, item_name):
            return getattr(mod, item_name)
    return default


def convert_operations(onnx_graph, opset_version, batch_dim=0, enable_pruning=True):
    """
    Convert onnx model operations. Yields onnx's operator_id, operator_name and
    converted pytorch operator.

    Parameters
    ----------
    onnx_graph: onnx.GraphProto
        Loaded onnx model's GraphProto.
    opset_version: int
        ONNX model's opset version.
    batch_dim: int
        Usually 0 for computer vision models and 1 for NLP models.
    enable_pruning: bool
        Track kept/pruned indices between different calls to forward pass.

    Returns
    -------
    iterator: (op_id, op_name, op)
    """
    weights = {tensor.name: tensor for tensor in onnx_graph.initializer}
    for i, node in enumerate(onnx_graph.node):
        # extract only useful inputs
        params = [weights[par_name] for par_name in node.input if par_name in weights]

        if node.op_type == "Add":
            # op = Add(feature_dim=batch_dim + 1)  # 0 for CV models and 1 for NLP
# ------------------------------------------------------------
# my layers
            if node.name[:5] == 'quant':
                # print('dequant_add')
                op = quan_add()
            else:
                # print('test add')
                op = Add(feature_dim=batch_dim + 1)  # 0 for CV models and 1 for NLP    
                # op = test_add()

# ------------------------------------------------------------
        elif node.op_type == "And":
            op = OperatorWrapper(torch.logical_and)
        elif node.op_type == "AveragePool":
            op = convert_layer(node, "AvgPool")
        elif node.op_type == "BatchNormalization":
            op = convert_batch_norm_layer(node, params=params)
        elif node.op_type == "Cast":
            op = Cast(**extract_attributes(node))
        elif node.op_type == "Ceil":
            op = OperatorWrapper(torch.ceil)
        elif node.op_type == "Clip":
            op = Clip(**extract_attributes(node))
        elif node.op_type == "Concat":
            op = partial(torch.cat, **extract_attributes(node))
        elif node.op_type == "Constant":
            op = Constant(**extract_attributes(node))
        elif node.op_type == "ConstantOfShape":
            op = ConstantOfShape(**extract_attributes(node))
        elif node.op_type == "Conv":
            op = convert_layer(node, "Conv", params)
        elif node.op_type == "ConvTranspose":
            op = convert_layer(node, "ConvTranspose", params)
        elif node.op_type == "Div":
            op = Div()
        elif node.op_type == "Elu":
            op = nn.ELU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Equal":
            op = OperatorWrapper(torch.eq)
        elif node.op_type == "Erf":
            op = OperatorWrapper(torch.erf)
        elif node.op_type == "Exp":
            op = OperatorWrapper(torch.exp)
        elif node.op_type == "Expand":
            op = Expand()
        elif node.op_type == "Flatten":
            op = Flatten(**extract_attributes(node))
            op.feature_dim = batch_dim + 1  # Necessary for transformers
        elif node.op_type == "Floor":
            op = OperatorWrapper(torch.floor)
        elif node.op_type == "Gather":
            op = Gather(**extract_attributes(node))
        elif node.op_type == "GatherND":
            op = GatherND(**extract_attributes(node))
        elif node.op_type == "Gemm":
            op = convert_linear_layer(node, params)
        elif node.op_type == "GlobalAveragePool":
            op = GlobalAveragePool()
        elif node.op_type == "Greater":
            op = OperatorWrapper(torch.greater)
        elif node.op_type == "Identity":
            op = nn.Identity()
        elif node.op_type == "InstanceNormalization":
            op = convert_instance_norm_layer(node, params=params)
        elif node.op_type == "LeakyRelu":
            op = nn.LeakyReLU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Less":
            op = OperatorWrapper(torch.less)
        elif node.op_type == "Log":
            op = OperatorWrapper(torch.log)
        elif node.op_type == "Loop":
            op = Loop(
                opset_version=opset_version,
                batch_dim=batch_dim,
                **extract_attributes(node),
            )
        elif node.op_type == "LSTM":
            op = convert_lstm_layer(node, weights)
        elif node.op_type == "MatMul":
            if params:
                weight = torch.from_numpy(numpy_helper.to_array(params[0]))
                op = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
                op.weight.data = weight.t()

                # check if next node Add to add bias
                next_node = onnx_graph.node[i + 1]
                next_params = [
                    weights[par_name]
                    for par_name in next_node.input
                    if par_name in weights
                ]
                if next_params and next_node.op_type == "Add":
                    bias = torch.from_numpy(numpy_helper.to_array(next_params[0]))
                    op.bias = nn.Parameter(bias)
                    node.output.pop()
                    node.output.extend(next_node.output)
                    onnx_graph.node.pop(i + 1)  # remove next node
            else:
                op = MatMul()
        elif node.op_type == "Max":
            op = OperatorWrapper(torch.max)
        elif node.op_type == "MaxPool":
            op = convert_layer(node, "MaxPool")
        elif node.op_type == "Min":
            op = OperatorWrapper(torch.min)
        elif node.op_type == "Mul":
            op = OperatorWrapper(torch.mul)
        elif node.op_type == "NonMaxSuppression":
            op = NonMaxSuppression(**extract_attributes(node))
        elif node.op_type == "Not":
            op = OperatorWrapper(torch.logical_not)
        elif node.op_type == "OneHot":
            op = OneHot(**extract_attributes(node))
        elif node.op_type == "Or":
            op = OperatorWrapper(torch.logical_or)
        elif node.op_type == "Pad":
            op = Pad(**extract_attributes(node))
        elif node.op_type == "Pow":
            op = OperatorWrapper(torch.pow)
        elif node.op_type == "PRelu":
            op = PRelu()
        elif node.op_type == "Range":
            op = Range()
        elif node.op_type == "Reciprocal":
            op = OperatorWrapper(torch.reciprocal)
        # elif node.op_type == "ReduceMax":
        #     kwargs = dict(keepdim=True)
        #     kwargs.update(extract_attributes(node))
        #     op = partial(torch.max, **kwargs)
        elif node.op_type == "ReduceMean":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.mean, **kwargs)
        # elif node.op_type == "ReduceMin":
        #     kwargs = dict(keepdim=True)
        #     kwargs.update(extract_attributes(node))
        #     op = partial(torch.min, **kwargs)
        elif node.op_type == "ReduceProd":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.prod, **kwargs)
        elif node.op_type == "ReduceSum":
            op = ReduceSum(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Relu":
            if node.name[:5] == "relu6":
                # print("find relu6")
                op = nn.ReLU6(inplace=False)
            else:
                # print("find relu")
                op = nn.ReLU(inplace=False)
            # op = nn.ReLU(inplace=False)
        elif node.op_type == "Reshape":
            shape = list(
                filter(lambda x: x.name == node.input[1], onnx_graph.initializer)
            )
            shape = np.copy(numpy_helper.to_array(shape[0])) if shape else None
            op = Reshape(enable_pruning, shape)
            # op = View(shape)
        elif node.op_type == "Resize":
            # print(extract_attributes(node))
            op = Resize(**extract_attributes(node))
        elif node.op_type == "Scatter":
            op = Scatter(**extract_attributes(node))
        elif node.op_type == "ScatterElements":
            op = ScatterElements(**extract_attributes(node))
        elif node.op_type == "ScatterND":
            op = ScatterND()
        elif node.op_type == "Shape":
            op = Shape()
        elif node.op_type == "Sigmoid":
            op = nn.Sigmoid()
        elif node.op_type == "Slice":
            op = Slice(**extract_attributes(node))
        elif node.op_type == "Softmax":
            kwargs = dict(dim=-1)
            kwargs.update(extract_attributes(node))
            op = nn.Softmax(**kwargs)
        elif node.op_type == "Softplus":
            op = nn.Softplus(beta=1)
        elif node.op_type == "Softsign":
            op = nn.Softsign()
        elif node.op_type == "Split":
            kwargs = extract_attributes(node)
            # if the split_size_or_sections is not in node attributes,
            # the number_of_splits becomes the number of node outputs
            if "split_size_or_sections" not in kwargs:
                kwargs["number_of_splits"] = len(node.output)
            op = Split(enable_pruning, **kwargs)
        elif node.op_type == "Sqrt":
            op = OperatorWrapper(torch.sqrt)
        elif node.op_type == "Squeeze":
            op = Squeeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Sub":
            op = OperatorWrapper(torch.sub)
        elif node.op_type == "Tanh":
            op = OperatorWrapper(torch.tanh)
        elif node.op_type == "ThresholdedRelu":
            op = ThresholdedRelu(**extract_attributes(node))
        elif node.op_type == "Tile":
            op = Tile()
        elif node.op_type == "TopK":
            op = TopK()
        elif node.op_type == "Transpose":
            op = Transpose(**extract_attributes(node))
        elif node.op_type == "Unsqueeze":
            op = Unsqueeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Upsample":
            op = Upsample(**extract_attributes(node))
        elif node.op_type == "Where":
            op = Where()
# ------------------------------------------------------------
# my layers
        elif node.op_type == "LpNormalization":
            op = Normalization()
            # error_sign['missing_op_sign'] = True
        elif node.op_type == "SpaceToDepth":
            # kwargs = extract_attributes(node)
            op = SpaceToDepth(**extract_attributes(node))
        elif node.op_type == "DepthToSpace":
            # kwargs = extract_attributes(node)
            op = DepthToSpace(**extract_attributes(node))
        elif node.op_type == "HardSwish":
            op = HardSwish()
        elif node.op_type == "Roll":
            # print(torch.from_numpy(numpy_helper.to_array(params[0])).tolist())
            # print(torch.from_numpy(numpy_helper.to_array(params[1])).tolist())
            if torch.numel(torch.from_numpy(numpy_helper.to_array(params[0]))) != 1:
                shift = tuple(torch.from_numpy(numpy_helper.to_array(params[0])).tolist())
                axis = tuple(torch.from_numpy(numpy_helper.to_array(params[1])).tolist())
            else:
                shift = torch.from_numpy(numpy_helper.to_array(params[0])).tolist()
                axis = torch.from_numpy(numpy_helper.to_array(params[1])).tolist()            
            kwargs = {}
            kwargs["shift"] = shift
            kwargs["axis"] = axis
            op = Roll(**kwargs)
        elif node.op_type == "Einsum":
            op = EinSum(**extract_attributes(node))
        elif node.op_type == "GreaterOrEqual":
            op = Great_Equal()
        # elif node.op_type == "GlobalMaxPool":
        #     op = GlobalMaxPool()
        elif node.op_type == "LessOrEqual":
            op = LessOrEqual()
        elif node.op_type == "ResizeNearestNeighbor":
            size = tuple(torch.from_numpy(numpy_helper.to_array(params[0])).tolist())
            op = ResizeNearestNeighbor(size, **extract_attributes(node))
        elif node.op_type == "ReduceMax":
            op = ReduceMax(**extract_attributes(node))
        elif node.op_type == "ReduceMin":
            op = ReduceMin(**extract_attributes(node))
        elif node.op_type == "LocalResponseNorm":
            op = LocalResponseNorm(**extract_attributes(node))
        elif node.op_type == "Dropout":
            op = Dropout(**extract_attributes(node))
        elif node.op_type == "ArgMax":
            op = Argmax(**extract_attributes(node))
        elif node.op_type == "Neg":
            op = OperatorWrapper(torch.neg)
        # elif node.op_type == "Sum":
        #     op = Add(feature_dim=batch_dim + 1)
            # continue
        # elif node.op_type == "DequantizeLinear":
        #     op = OperatorWrapper(torch.dequantize)
# ------------------------------------------------------------
        else:
            op = getattr(torch, node.op_type.lower(), None)
            if op is None:
                raise NotImplementedError(
                    "Conversion not implemented for op_type={}.".format(node.op_type)
                )
            else:
                print(
                    "Automatic inference of operator: {}".format(node.op_type.lower())
                )

        op_name = "{}_{}".format(node.op_type, node.output[0])
        op_id = node.output[0]
        yield op_id, op_name, op
