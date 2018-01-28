# Efficient-ShuffleNet
This repository implements a ShuffleNet prototxt generator and a more efficient channel shuffle implementation, gives a benchmark of forward backward time of shufflenet 1x_g3 on Pascal Titan X.

farmingyard's implementation(https://github.com/farmingyard/ShuffleNet) of shufflechannel layer is not efficient enough as it use each thread to do a single channel shuffle operation, if input size increase, the overhead of shufflechannel layer would be  large. Here a permute layer of weiliu's implementation [Permute_layer](https://github.com/BVLC/caffe/commit/b68695db42aa79e874296071927536363fe1efbf?diff=unified) is used for more efficient channel shuffle operation.

# Benchmark forward backward time
Test on TiTan X Pascal

shufflenet_0.25×_g3 input = 50*224*224

+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 28%   50C    P2    80W / 250W |   2363MiB / 12189MiB |     42%   E. Process |

|I0128 13:27:48.656941 32473 caffe.cpp:460] --------Total Time Convolution: 7.230897172910
|I0128 13:27:48.656952 32473 caffe.cpp:461] --------Total Time BatchNorm: 10.60147172910
|I0128 13:27:48.656965 32473 caffe.cpp:462] --------Total Time Scale: 2.980527172910
|I0128 13:27:48.656972 32473 caffe.cpp:463] --------Total Time ReLU: 1.21517172910
|I0128 13:27:48.656978 32473 caffe.cpp:464] --------Total Time Pooling: 0.5667077172910
|I0128 13:27:48.656985 32473 caffe.cpp:465] --------Total Time Permute: 1.536877172910
|I0128 13:27:48.656989 32473 caffe.cpp:466] --------Total Time ShuffleChannel: 0
|I0128 13:27:48.656996 32473 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 0.8519497172910
|I0128 13:27:48.657001 32473 caffe.cpp:468] --------Total Time Split: 0.02986247172910
|I0128 13:27:48.657007 32473 caffe.cpp:469] --------Total Time Eltwise: 1.660027172910
|I0128 13:27:48.657013 32473 caffe.cpp:470] --------Total Time FullyConnected: 0.05816967172910
|I0128 13:27:48.657019 32473 caffe.cpp:471] --------Total Time SoftMax: 0.01597127172910
|I0128 13:27:48.657024 32473 caffe.cpp:475] Average Forward pass: 26.9364 ms.
|I0128 13:27:48.657032 32473 caffe.cpp:477] Average Backward pass: 57.718 ms.

shufflenet_0.25×_g3 input = 1*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 31%   53C    P2    84W / 250W |    487MiB / 12189MiB |     48%   E. Process |

I0128 13:28:39.061727 32607 caffe.cpp:460] --------Total Time Convolution: 1.58957172910
I0128 13:28:39.061738 32607 caffe.cpp:461] --------Total Time BatchNorm: 3.452047172910
I0128 13:28:39.061743 32607 caffe.cpp:462] --------Total Time Scale: 0.6270757172910
I0128 13:28:39.061749 32607 caffe.cpp:463] --------Total Time ReLU: 0.2694377172910
I0128 13:28:39.061755 32607 caffe.cpp:464] --------Total Time Pooling: 0.06235847172910
I0128 13:28:39.061761 32607 caffe.cpp:465] --------Total Time Permute: 0.4028937172910
I0128 13:28:39.061767 32607 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:28:39.061774 32607 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 0.2428327172910
I0128 13:28:39.061779 32607 caffe.cpp:468] --------Total Time Split: 0.02669447172910
I0128 13:28:39.061784 32607 caffe.cpp:469] --------Total Time Eltwise: 0.1845067172910
I0128 13:28:39.061790 32607 caffe.cpp:470] --------Total Time FullyConnected: 0.01445447172910
I0128 13:28:39.061796 32607 caffe.cpp:471] --------Total Time SoftMax: 0.01245767172910
I0128 13:28:39.061802 32607 caffe.cpp:475] Average Forward pass: 6.97196 ms.
I0128 13:28:39.061807 32607 caffe.cpp:477] Average Backward pass: 7.40818 ms.

shufflenet_0.5×_g3 input = 50*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 32%   58C    P2    82W / 250W |   3975MiB / 12189MiB |     13%   E. Process |

I0128 13:23:09.701665 31934 caffe.cpp:460] --------Total Time Convolution: 8.032357172910
I0128 13:23:09.701678 31934 caffe.cpp:461] --------Total Time BatchNorm: 16.3297172910
I0128 13:23:09.701685 31934 caffe.cpp:462] --------Total Time Scale: 4.332697172910
I0128 13:23:09.701691 31934 caffe.cpp:463] --------Total Time ReLU: 1.893627172910
I0128 13:23:09.701699 31934 caffe.cpp:464] --------Total Time Pooling: 0.6690057172910
I0128 13:23:09.701704 31934 caffe.cpp:465] --------Total Time Permute: 2.433197172910
I0128 13:23:09.701711 31934 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:23:09.701717 31934 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 1.312777172910
I0128 13:23:09.701722 31934 caffe.cpp:468] --------Total Time Split: 0.03042887172910
I0128 13:23:09.701730 31934 caffe.cpp:469] --------Total Time Eltwise: 3.133847172910
I0128 13:23:09.701735 31934 caffe.cpp:470] --------Total Time FullyConnected: 0.0784487172910
I0128 13:23:09.701741 31934 caffe.cpp:471] --------Total Time SoftMax: 0.01666567172910
I0128 13:23:09.701747 31934 caffe.cpp:475] Average Forward pass: 38.543 ms.
I0128 13:23:09.701753 31934 caffe.cpp:477] Average Backward pass: 77.4036 ms.

shufflenet_0.5×_g3 input = 1*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 29%   46C    P2    86W / 250W |    519MiB / 12189MiB |     49%   E. Process |

I0128 13:22:03.776329 31793 caffe.cpp:460] --------Total Time Convolution: 1.67177172910
I0128 13:22:03.776341 31793 caffe.cpp:461] --------Total Time BatchNorm: 3.54417172910
I0128 13:22:03.776348 31793 caffe.cpp:462] --------Total Time Scale: 0.6465547172910
I0128 13:22:03.776355 31793 caffe.cpp:463] --------Total Time ReLU: 0.2912337172910
I0128 13:22:03.776361 31793 caffe.cpp:464] --------Total Time Pooling: 0.06677667172910
I0128 13:22:03.776368 31793 caffe.cpp:465] --------Total Time Permute: 0.4296187172910
I0128 13:22:03.776376 31793 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:22:03.776383 31793 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 0.2647417172910
I0128 13:22:03.776391 31793 caffe.cpp:468] --------Total Time Split: 0.02743497172910
I0128 13:22:03.776396 31793 caffe.cpp:469] --------Total Time Eltwise: 0.199887172910
I0128 13:22:03.776409 31793 caffe.cpp:470] --------Total Time FullyConnected: 0.02097067172910
I0128 13:22:03.776417 31793 caffe.cpp:471] --------Total Time SoftMax: 0.01338437172910
I0128 13:22:03.776424 31793 caffe.cpp:475] Average Forward pass: 7.26749 ms.
I0128 13:22:03.776430 31793 caffe.cpp:477] Average Backward pass: 6.9241 ms.

shufflenet_1×_g3 input = 50*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 30%   56C    P2    81W / 250W |   6671MiB / 12189MiB |     90%   E. Process |

I0128 13:20:12.433203 31561 caffe.cpp:460] --------Total Time Convolution: 10.51457172910
I0128 13:20:12.433221 31561 caffe.cpp:461] --------Total Time BatchNorm: 27.93637172910
I0128 13:20:12.433233 31561 caffe.cpp:462] --------Total Time Scale: 7.385327172910
I0128 13:20:12.433244 31561 caffe.cpp:463] --------Total Time ReLU: 3.34637172910
I0128 13:20:12.433255 31561 caffe.cpp:464] --------Total Time Pooling: 0.8739747172910
I0128 13:20:12.433265 31561 caffe.cpp:465] --------Total Time Permute: 4.639487172910
I0128 13:20:12.433275 31561 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:20:12.433286 31561 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 2.168897172910
I0128 13:20:12.433297 31561 caffe.cpp:468] --------Total Time Split: 0.02919367172910
I0128 13:20:12.433307 31561 caffe.cpp:469] --------Total Time Eltwise: 6.150137172910
I0128 13:20:12.433318 31561 caffe.cpp:470] --------Total Time FullyConnected: 0.1263947172910
I0128 13:20:12.433328 31561 caffe.cpp:471] --------Total Time SoftMax: 0.0165127172910
I0128 13:20:12.433339 31561 caffe.cpp:475] Average Forward pass: 63.6526 ms.
I0128 13:20:12.433351 31561 caffe.cpp:477] Average Backward pass: 117.361 ms.

shufflenet_1×_g3 input = 1*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 29%   52C    P2    92W / 250W |    589MiB / 12189MiB |     53%   E. Process |

I0128 13:21:23.503549 31711 caffe.cpp:460] --------Total Time Convolution: 1.935327172910
I0128 13:21:23.503561 31711 caffe.cpp:461] --------Total Time BatchNorm: 3.566057172910
I0128 13:21:23.503567 31711 caffe.cpp:462] --------Total Time Scale: 0.6873167172910
I0128 13:21:23.503573 31711 caffe.cpp:463] --------Total Time ReLU: 0.2990767172910
I0128 13:21:23.503579 31711 caffe.cpp:464] --------Total Time Pooling: 0.06651687172910
I0128 13:21:23.503585 31711 caffe.cpp:465] --------Total Time Permute: 0.4606727172910
I0128 13:21:23.503592 31711 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:21:23.503597 31711 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 0.2802527172910
I0128 13:21:23.503603 31711 caffe.cpp:468] --------Total Time Split: 0.02756837172910
I0128 13:21:23.503618 31711 caffe.cpp:469] --------Total Time Eltwise: 0.2245397172910
I0128 13:21:23.503625 31711 caffe.cpp:470] --------Total Time FullyConnected: 0.03434377172910
I0128 13:21:23.503631 31711 caffe.cpp:471] --------Total Time SoftMax: 0.01252267172910
I0128 13:21:23.503638 31711 caffe.cpp:475] Average Forward pass: 7.68768 ms.
I0128 13:21:23.503643 31711 caffe.cpp:477] Average Backward pass: 7.65132 ms.

shufflenet_2×_g3 input = 50*224*224

+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 35%   63C    P2   197W / 250W |  11871MiB / 12189MiB |     96%   E. Process |

I0128 13:25:44.941494 32265 caffe.cpp:460] --------Total Time Convolution: 16.63937172910
I0128 13:25:44.941505 32265 caffe.cpp:461] --------Total Time BatchNorm: 49.94187172910
I0128 13:25:44.941511 32265 caffe.cpp:462] --------Total Time Scale: 13.3097172910
I0128 13:25:44.941519 32265 caffe.cpp:463] --------Total Time ReLU: 6.136867172910
I0128 13:25:44.941524 32265 caffe.cpp:464] --------Total Time Pooling: 1.248567172910
I0128 13:25:44.941529 32265 caffe.cpp:465] --------Total Time Permute: 8.628387172910
I0128 13:25:44.941535 32265 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:25:44.941541 32265 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 3.951857172910
I0128 13:25:44.941547 32265 caffe.cpp:468] --------Total Time Split: 0.02986247172910
I0128 13:25:44.941553 32265 caffe.cpp:469] --------Total Time Eltwise: 12.04857172910
I0128 13:25:44.941558 32265 caffe.cpp:470] --------Total Time FullyConnected: 0.1775337172910
I0128 13:25:44.941565 32265 caffe.cpp:471] --------Total Time SoftMax: 0.01610247172910
I0128 13:25:44.941571 32265 caffe.cpp:475] Average Forward pass: 112.96 ms.
I0128 13:25:44.941576 32265 caffe.cpp:477] Average Backward pass: 186.387 ms.

shufflenet_2×_g3 input = 1*224*224
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:09:00.0     Off |                  N/A |
| 33%   55C    P2   104W / 250W |    741MiB / 12189MiB |     58%   E. Process |

I0128 13:24:08.907522 32058 caffe.cpp:460] --------Total Time Convolution: 2.353567172910
I0128 13:24:08.907533 32058 caffe.cpp:461] --------Total Time BatchNorm: 3.759117172910
I0128 13:24:08.907539 32058 caffe.cpp:462] --------Total Time Scale: 0.7835167172910
I0128 13:24:08.907546 32058 caffe.cpp:463] --------Total Time ReLU: 0.3336697172910
I0128 13:24:08.907552 32058 caffe.cpp:464] --------Total Time Pooling: 0.07047467172910
I0128 13:24:08.907557 32058 caffe.cpp:465] --------Total Time Permute: 0.5566337172910
I0128 13:24:08.907570 32058 caffe.cpp:466] --------Total Time ShuffleChannel: 0
I0128 13:24:08.907577 32058 caffe.cpp:467] --------Total Time ConvolutionDepthwise: 0.3202857172910
I0128 13:24:08.907583 32058 caffe.cpp:468] --------Total Time Split: 0.02727337172910
I0128 13:24:08.907589 32058 caffe.cpp:469] --------Total Time Eltwise: 0.2922717172910
I0128 13:24:08.907595 32058 caffe.cpp:470] --------Total Time FullyConnected: 0.05242627172910
I0128 13:24:08.907600 32058 caffe.cpp:471] --------Total Time SoftMax: 0.0125047172910
I0128 13:24:08.907606 32058 caffe.cpp:475] Average Forward pass: 8.65975 ms.
I0128 13:24:08.907613 32058 caffe.cpp:477] Average Backward pass: 9.2066 ms.

# Steps to Use
if you don't have permute layer yet, then put `permute_layer.hpp` into `CAFFE_ROOT/include/caffe/layers/`, put `permute_layer.cpp` and `permute_layer.cu` into `CAFFE_ROOT/src/caffe/layers/`
if you don't have depthWise convolution layer yet, then do the same as permute layer for source files from [DepthWise Convolution layer](https://github.com/farmingyard/caffe-mobilenet)

By default caffe 


and then just do `make` under `CAFFE_ROOT/`
