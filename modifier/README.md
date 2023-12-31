## Rules for Pruning and Translation

We use the Pruning List and Translation List in the Pruning Module and Translation Module in the proposed method, respectively. 
The rule lists for the Pruning Module and the Translation Module.
Note that this list may not be complete for addressing all transforming errors in TFLite to differentiable (debuggable) models.
Our prototype tool can solve most of the cases. 
We use if conditions in PyThon to detect the potential mismatch problems that are in our list.
You can modify (or add) the conditions and corresponding operations (e.g., removing in the Pruning Module, Translation in the Translation Module) in our codes if you have errors when using our tool or if the DL libraries (e.g., TFLite, ONNX) have been updated. 

Left: pruning List. Right: translation List and equivalent combination list. 
The elements of the two parts of the Translation Module are ordered by the co-relation. 
For example, the **DynamicQuantizeLinear** operator can be divided into two separate operators **(div, add)**. 
Note that one operator may cause two different problems.

| Pruning List | Translation List |
| -------- | ------- |
| DynamicQuantizeLinear | DynamicQuantizeLinear - (max, min, div, add) |
| DequantizeLinear | QuantizeLinear - (div, add) |
| Gemm | DequantizeLinear - (Sub, Mul) |
| QuantizeLinear | SpaceToDepth - (reshape, transpose, reshape) |
| Transpose | DepthToSpace - (reshape, transpose, reshape) |
| Reshape | ConvInteger - (Sub, Conv) |
| Unsqueeze | MatMulInteger - (Sub, MatMul) |
| Squeeze | GlobalMaxPool - (Flatten, Max) |
| bitwise |  |
