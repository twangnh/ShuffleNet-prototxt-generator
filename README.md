# Efficient-ShuffleNet
This repository implements a ShuffleNet prototxt generator and a more efficient channel shuffle implementation, gives a benchmark of forward backward time of shufflenet 1x_g3 on Pascal Titan X.

farmingyard's implementation(https://github.com/farmingyard/ShuffleNet) of shufflechannel layer is not efficient enough as it use each thread to do a full channel shuffle operation. And if input image size(i.e., height and width) increase, the overhead of shufflechannel layer would be large. Here a permute layer of weiliu's implementation [Permute_layer](https://github.com/BVLC/caffe/commit/b68695db42aa79e874296071927536363fe1efbf?diff=unified) is used for more efficient channel shuffle operation.

# Benchmark forward backward time
Test on TiTan X Pascal


![](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_0.25x_g3.png)

![](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_0.5x_g3.png)

![](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_1x_g3.png)

![](https://github.com/MrWanter/Efficient-ShuffleNet/blob/master/image/shufflenet_2x_g3.png)

# Steps to Use
if you don't have permute layer yet, then put `permute_layer.hpp` into `CAFFE_ROOT/include/caffe/layers/`, put `permute_layer.cpp` and `permute_layer.cu` into `CAFFE_ROOT/src/caffe/layers/`
if you don't have depthWise convolution layer yet, then do the same as permute layer for source files from [DepthWise Convolution layer](https://github.com/farmingyard/caffe-mobilenet)

then you can safely change  [the line](https://github.com/BVLC/caffe/blob/bb4ffa4d440e8a9c452c410ad9db2ed7137c9f7d/include/caffe/blob.hpp#L140) in caffe sourece to `CHECK_LE(num_axes(), 5)`

and then just do `make` under `CAFFE_ROOT/`
