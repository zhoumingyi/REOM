import torch.nn.functional as F
import torch

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        # --------------------------------------------
        # my code
        shape = torch.tensor(list(pads))
        # print("Pad shape:", shape)
        # if shape.numel() == 8:
        #     index = [3,2,1,0]            
        # elif shape.numel() == 4:
        #     index = [1,0]
        # elif shape.numel() == 6:
        #     index = [2,1,0]
        # else:
        #     raise Exception('Ohhhhhhh, Pad shape is wrong')
        index = [(i-1) for i in range(int(shape.numel()) // 2, 0, -1)]
        shape = shape.reshape(2, shape.numel()//2).T
        shape = shape[index].flatten()
        out = F.pad(input, shape.tolist(), mode=self.mode, value=value)
        # --------------------------------------------
        # out = F.pad(input, list(pads), mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)