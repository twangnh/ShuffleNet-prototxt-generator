import os
import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser

CAFFE_ROOT = osp.join(osp.dirname(__file__), 'caffe-fast-rcnn')

if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
from caffe.proto import caffe_pb2

def _get_param(num_param):
    if num_param == 1:
        # only weight
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1
        param.decay_mult = 1
        return [param]
    elif num_param == 2:
        # weight and bias
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1
        param_w.decay_mult = 1
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 2
        param_b.decay_mult = 0
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))


def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer


def BN(data, name=None):
    # BatchNorm
    layers = []
    bn_layer = caffe_pb2.LayerParameter()
    bn_layer.name = name + '_bn'
    bn_layer.type = 'BatchNorm'
    bn_layer.bottom.append(data)
    bn_layer.top.append(name + 'bn_data')
    layers.append(bn_layer)
    # Scale
    scale_layer = caffe_pb2.LayerParameter()
    scale_layer.name = name + '_scale'
    scale_layer.type = 'Scale'
    scale_layer.bottom.append(name + 'bn_data')
    scale_layer.top.append(name + 'bn_data')
    scale_layer.scale_param.filler.value = 1
    scale_layer.scale_param.bias_term = True
    scale_layer.scale_param.bias_filler.value = 0
    layers.append(scale_layer)
    
    return layers

def AC(data, name = None):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + '_relu'
    layer.type = 'ReLU'
    layer.bottom.append(data)
    layer.top.append(name + 'ac_data')

    return layer

def BN_AC(data, name=None):
    # BatchNorm
    layers = []
    bn_layer = caffe_pb2.LayerParameter()
    bn_layer.name = name + '_bn'
    bn_layer.type = 'BatchNorm'
    bn_layer.bottom.append(data)
    bn_layer.top.append(name + 'bn_ac_data')
    layers.append(bn_layer)
    # Scale
    scale_layer = caffe_pb2.LayerParameter()
    scale_layer.name = name + '_scale'
    scale_layer.type = 'Scale'
    scale_layer.bottom.append(name + 'bn_ac_data')
    scale_layer.top.append(name + 'bn_ac_data')
    scale_layer.scale_param.filler.value = 1
    scale_layer.scale_param.bias_term = True
    scale_layer.scale_param.bias_filler.value = 0
    layers.append(scale_layer)
    # Relu
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_relu'
    relu_layer.type = 'ReLU'
    relu_layer.bottom.append(name + 'bn_ac_data')
    relu_layer.top.append(name + 'bn_ac_data')
    layers.append(relu_layer)

    return layers

def Conv(data=None,  num_filter=32, num_group = 1,  kernel=(5, 5), name='conv1_1', pad=(2,2), stride=(1,1)):

    Conv_layer = caffe_pb2.LayerParameter()
    Conv_layer.name = name
    Conv_layer.type = 'Convolution'
    Conv_layer.bottom.append(data)
    Conv_layer.top.append(name + 'Conv_data')
    Conv_layer.convolution_param.num_output = num_filter
    Conv_layer.convolution_param.kernel_h = kernel[0]
    Conv_layer.convolution_param.kernel_w = kernel[1]
    Conv_layer.convolution_param.stride_h = stride[0]
    Conv_layer.convolution_param.stride_w = stride[1]
    Conv_layer.convolution_param.pad_h = pad[0]
    Conv_layer.convolution_param.pad_w = pad[1]
    Conv_layer.convolution_param.group = num_group

    Conv_layer.convolution_param.weight_filler.type = 'msra'
    Conv_layer.convolution_param.bias_term = False
    Conv_layer.param.extend(_get_param(1))

    return Conv_layer

def Conv_DepthWise(data=None,  kernel=(3, 3), name='conv_depth_wise', pad=(0,0), stride=(1,1)):
    Conv_DepthWise_layer = caffe_pb2.LayerParameter()
    Conv_DepthWise_layer.name = name 
    Conv_DepthWise_layer.type = 'ConvolutionDepthwise'
    Conv_DepthWise_layer.bottom.append(data)
    Conv_DepthWise_layer.top.append(name + 'Conv_data')
    Conv_DepthWise_layer.convolution_param.kernel_h = kernel[0]
    Conv_DepthWise_layer.convolution_param.kernel_w = kernel[1]
    Conv_DepthWise_layer.convolution_param.stride_h = stride[0]
    Conv_DepthWise_layer.convolution_param.stride_w = stride[1]
    Conv_DepthWise_layer.convolution_param.pad_h = pad[0]
    Conv_DepthWise_layer.convolution_param.pad_w = pad[1]

    Conv_DepthWise_layer.convolution_param.weight_filler.type = 'msra'
    Conv_DepthWise_layer.convolution_param.bias_term = False
    Conv_DepthWise_layer.param.extend(_get_param(1))

    return Conv_DepthWise_layer


def Pool(data, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(1,1), name="pool1", re_interface = False):

    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Pooling'
    layer.bottom.append(data)
    layer.top.append(name + 'pool_data')
    if pool_type == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pool_type == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    else:
        raise ValueError("Unknown pooling method {}".format(pooling_method))
    layer.pooling_param.kernel_h = kernel[0]
    layer.pooling_param.kernel_w = kernel[1]
    layer.pooling_param.stride_h = stride[0]
    layer.pooling_param.stride_w = stride[1]
    layer.pooling_param.pad_h = pad[0]
    layer.pooling_param.pad_w = pad[1]
    if re_interface:
        return name + 'pool_data', layer
    else:
        return layer

def Flatten(data=None, name='flatten'):

    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Flatten'
    layer.bottom.append(data)
    layer.top.append(name)
    
    return layer


def FullyConnected(data, num_hidden, name='fc6'):

    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.append(data)
    layer.top.append(name)
    layer.inner_product_param.num_output = num_hidden
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_filler.type = 'msra'

    return layer

def SoftmaxOutput(data,  name='softmax'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Softmax'
    layer.bottom.append(data)
    layer.top.append(name)

    return layer


def ElementWiseSum(bottoms, name, re_interface=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.append(name)
    if re_interface:
        return name, layer
    else:
        return layer

def Concate(bottoms, name, re_interface=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Concat'
    layer.bottom.extend(bottoms)
    layer.top.append(name)
    if re_interface:
        return name, layer
    else:
        return layer

def Shuffle_Channel(data, num_group = 3, name= None):
    layers = []
    Reshape_pre_layer = caffe_pb2.LayerParameter()
    Reshape_pre_layer.name = name + '_reshape_pre_layer'
    Reshape_pre_layer.type = 'Reshape'
    Reshape_pre_layer.bottom.append(data)
    Reshape_pre_layer.top.append(name + 'pre_shuffle_data')
    Reshape_pre_layer.reshape_param.shape.dim.extend([num_group, -1])
    Reshape_pre_layer.reshape_param.axis = 1
    Reshape_pre_layer.reshape_param.num_axes = 1
    layers.append(Reshape_pre_layer)

    Permute_layer = caffe_pb2.LayerParameter()
    Permute_layer.name  = name + '_Permute_layer'
    Permute_layer.type = 'Permute'
    Permute_layer.bottom.append(name + 'pre_shuffle_data')
    Permute_layer.top.append(name + 'shuffle_data')
    Permute_layer.permute_param.order.extend([0, 2, 1, 3, 4])
    layers.append(Permute_layer)

    Reshape_post_layer = caffe_pb2.LayerParameter()
    Reshape_post_layer.name = name + '_reshape_post_layer'
    Reshape_post_layer.type = 'Reshape'
    Reshape_post_layer.bottom.append(name + 'shuffle_data')
    Reshape_post_layer.top.append(name + 'post_shuffle_data')
    Reshape_post_layer.reshape_param.shape.dim.append(-1)
    Reshape_post_layer.reshape_param.axis = 1
    Reshape_post_layer.reshape_param.num_axes = 2
    layers.append(Reshape_post_layer)
    
    return layers

def ShuffleNet_Unit_Factory(data, G, num_out, name, num_inner=None, _type='normal'):

    layers = []
    if _type == 'normal':

        #default inner channel number:
        num_inner = num_out/4

        layers.append(Conv(data=data,  num_filter = num_inner, num_group = G,  kernel=(1, 1), name= name + 'GConv_1', pad=(0,0), stride=(1,1)))
        layers.extend(BN_AC(data = layers[-1].top[0], name = name + 'GConv_1'))
        layers.extend(Shuffle_Channel(data = layers[-1].top[0], num_group = G, name= name + 'Shuffle'))
        layers.append(Conv_DepthWise(data = layers[-1].top[0],  kernel=(3, 3), name= name + 'conv_depth_wise', pad=(1,1), stride=(1,1)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'conv_depth_wise'))
        layers.append(Conv(data = layers[-1].top[0], num_filter = num_out, num_group = G,  kernel=(1, 1), name= name + 'GConv_2', pad=(0,0), stride=(1,1)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'GConv_2'))
        layers.append(ElementWiseSum([data, layers[-1].top[0]], name = name + 'Add'))
        layers.append(AC(data = layers[-1].top[0], name = name + 'Add'))
        
    elif _type == 'down':

        #the first unit of stage 3-4 will concat to get the num_out channel, so half it
        num_out/=2
        num_inner = num_out/4

        pool1, layer = Pool(data = data, pool_type = "ave", kernel=(3, 3), pad=(0,0), stride=(2,2), name= name + "pool1", re_interface = True)
        layers.append(layer)

        layers.append(Conv(data=data,  num_filter = num_inner, num_group = G,  kernel=(1, 1), name= name + 'GConv_1', pad=(0,0), stride=(1,1)))
        layers.extend(BN_AC(data = layers[-1].top[0], name = name + 'GConv_1'))
        layers.extend(Shuffle_Channel(data = layers[-1].top[0], num_group = G, name= name + 'Shuffle'))
        layers.append(Conv_DepthWise(data = layers[-1].top[0],  kernel=(3, 3), name= name + 'conv_depth_wise', pad=(1,1), stride=(2,2)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'conv_depth_wise'))
        layers.append(Conv(data = layers[-1].top[0], num_filter = num_out, num_group = G,  kernel=(1, 1), name= name + 'GConv_2', pad=(0,0), stride=(1,1)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'GConv_2'))

        layers.append(Concate([pool1, layers[-1].top[0]], name = name + 'Concate'))
        layers.append(AC(data = layers[-1].top[0], name = name + 'Concate'))

    elif _type == 'first_point_wise_no_group':

        num_inner = (num_out-24)/4

        pool1, layer = Pool(data = data, pool_type = "ave", kernel=(3, 3), pad=(0,0), stride=(2,2), name= name + "pool1", re_interface = True)
        layers.append(layer)

        layers.append(Conv(data=data,  num_filter = num_inner, num_group = 1,  kernel=(1, 1), name= name + 'GConv_1', pad=(0,0), stride=(1,1)))
        layers.extend(BN_AC(data = layers[-1].top[0], name = name + 'GConv_1'))
        layers.append(Conv_DepthWise(data = layers[-1].top[0],  kernel=(3, 3), name= name + 'conv_depth_wise', pad=(1,1), stride=(2,2)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'conv_depth_wise'))
        layers.append(Conv(data = layers[-1].top[0], num_filter = num_out - 24, num_group = 1,  kernel=(1, 1), name= name + 'GConv_2', pad=(0,0), stride=(1,1)))
        layers.extend(BN(data = layers[-1].top[0], name = name + 'GConv_2'))

        layers.append(Concate([pool1, layers[-1].top[0]], name = name + 'Concate'))
        layers.append(AC(data = layers[-1].top[0], name = name + 'Concate'))
    return layers
#settings----------------------------
g1   =    { 2: 144, \
            3: 288, \
            4: 576,  }

g2   =    { 2: 200, \
            3: 400, \
            4: 800,  }

g3   =    { 2: 240, \
            3: 480, \
            4: 960,  }

g4   =    { 2: 272, \
            3: 544, \
            4: 1088,  }

g8   =    { 2: 384, \
            3: 768, \
            4: 1536,  }

group_channel = { 1: g1, \
                  2: g2, \
                  3: g3, \
                  4: g4, \
                  8: g8, }

# add 2 because start from index 2 so 0 and 1 are ignored
stage_repeate  = {  2: 3+2, \
                    3: 7+2, \
                    4: 3+2,  }


def get_before_pool_shufflenet(scale_factor = 1, group = 3):
    input_data = 'data_input'
    layers = []
    layers.append(Conv(data=input_data,  num_filter = 24, num_group = 1,  kernel=(3, 3), name='conv1', pad=(1,1), stride=(2,2)))
    layers.extend(BN_AC(data = layers[-1].top[0], name = 'conv1'))
    layers.append(Pool(data=layers[-1].top[0], pool_type="max", kernel=(3, 3), pad=(0,0), stride=(2,2), name="pool1"))
    
    stage = 2
    layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage2_Unit1', _type='first_point_wise_no_group'))
    for i in range(2, stage_repeate[stage]):
        layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage2_Unit%d'% i, _type='normal'))

    stage = 3 
    layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage3_Unit1', _type='down'))
    for i in range(2, stage_repeate[stage]):
        layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage3_Unit%d'% i, _type='normal'))
        
    stage = 4
    layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage4_Unit1', _type='down'))
    for i in range(2, stage_repeate[stage]):
        layers.extend(ShuffleNet_Unit_Factory(layers[-1].top[0], G = group, num_out = int(group_channel[group][stage]*scale_factor), name = 'stage4_Unit%d'% i, _type='normal'))

    return layers

def get_before_pool(**kwargs):
    if not kwargs.get('net_name'):
        #default ShuffleNet
        return get_before_pool_shufflenet(kwargs.get('scale_factor'), kwargs.get('group'))
    elif kwargs['net_name'] == 'shufflenet':
        return get_before_pool_shufflenet(kwargs.get('scale_factor'), kwargs.get('group'))
    elif kwargs['net_name'] == 'mobilenet':
        raise NotImplementedError
    else:
        raise NotImplementedError

def get_linear(num_classes = 1000, **kwargs):
    layers = []
    layers.extend(get_before_pool(**kwargs))
    layers.append(Pool(layers[-1].top[0], pool_type="ave", kernel=(7, 7), stride=(1,1), pad=(0,0), name="pool5"))
    layers.append(Flatten(layers[-1].top[0], name='flatten'))
    layers.append(FullyConnected(layers[-1].top[0], num_hidden=num_classes, name='fc6'))
    return layers
 
def get_model(num_classes = 1000, **kwargs):
    model = caffe_pb2.NetParameter()
    model.name = '{net_name}_{scale_factor}_g{group}'.format(net_name = kwargs.get('net_name'), scale_factor = kwargs.get('scale_factor'), group = kwargs.get('group'))
    layers = []
    layers.extend(get_linear(num_classes, **kwargs))
    layers.append(SoftmaxOutput(layers[-1].top[0],  name='softmax'))
    model.layer.extend(layers)
    return model


def main(args):
    model = get_model(net_name = args.net_name, group = args.group, scale_factor = args.scale_factor)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__),'{}_{}_g{}.prototxt'.format(args.net_name, args.scale_factor, args.group))
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--net_name', type = str, default = 'shufflenet')
    parser.add_argument('--group', type=int, default=3,
                        choices=[1, 2, 3, 4, 8])
    parser.add_argument('--scale_factor', type=float, default=1.,
                        choices=[2.,1.5,1.,0.5,0.25])
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    main(args)
