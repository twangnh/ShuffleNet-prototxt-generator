# Efficient-ShuffleNet
This repository implements a ShuffleNet prototxt generator and a more efficient channel shuffle implementation, gives a benchmark of forward backward time of shufflenet 1x_g3 on Pascal Titan X.

farmingyard's implementation(https://github.com/farmingyard/ShuffleNet) of shufflechannel layer is not efficient enough as it use each thread to do a single channel shuffle operation, if input size increase, the overhead of shufflechannel layer would be  large. Here a permute layer of weiliu's implementation [Permute_layer](https://github.com/BVLC/caffe/commit/b68695db42aa79e874296071927536363fe1efbf?diff=unified) is used for more efficient channel shuffle operation.

# Benchmark forward backward time
Test on TiTan X Pascal

shufflenet_0.25×_g3
![shufflenet_0.25×_g3](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_0.25×_g3.png)
shufflenet_0.5×_g3
![shufflenet_0.25×_g3](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_0.25x_g3.png)
shufflenet_1×_g3
![shufflenet_1×_g3](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/shufflenet_1×_g3.png)
shufflenet_0.25×_g3
![shufflenet_2×_g3](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/shufflenet_2×_g3.png)

# Steps to Use
if you don't have permute layer yet, then put `permute_layer.hpp` into `CAFFE_ROOT/include/caffe/layers/`, put `permute_layer.cpp` and `permute_layer.cu` into `CAFFE_ROOT/src/caffe/layers/`
if you don't have depthWise convolution layer yet, then do the same as permute layer for source files from [DepthWise Convolution layer](https://github.com/farmingyard/caffe-mobilenet)

By default caffe 


and then just do `make` under `CAFFE_ROOT/`
