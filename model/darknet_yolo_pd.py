#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-11 16:31:57
#   Description : pytorch_yolov3
#
#================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

def conv2d_unit(x, filters, kernels, stride, padding, name, is_test, trainable):
    x = fluid.layers.conv2d(
        input=x,
        num_filters=filters,
        filter_size=kernels,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=name + ".conv.weights", trainable=trainable),
        bias_attr=False)
    bn_name = name + ".bn"
    if not trainable:   # 冻结层时（即trainable=False），bn的均值、标准差也还是会变化，只有设置is_test=True才保证不变
        is_test = True
    x = fluid.layers.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=bn_name + '.scale'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=bn_name + '.offset'),
        moving_mean_name=bn_name + '.mean',
        moving_variance_name=bn_name + '.var')
    x = fluid.layers.leaky_relu(x, alpha=0.1)
    return x

def residual_block(inputs, filters, conv_start_idx, is_test, trainable):
    x = conv2d_unit(inputs, filters, (1, 1), stride=1, padding=0, name='conv%.2d'% conv_start_idx, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, 2 * filters, (3, 3), stride=1, padding=1, name='conv%.2d'% (conv_start_idx+1), is_test=is_test, trainable=trainable)
    x = fluid.layers.elementwise_add(x=inputs, y=x, act=None)
    return x

def stack_residual_block(inputs, filters, n, conv_start_idx, is_test, trainable):
    x = residual_block(inputs, filters, conv_start_idx, is_test, trainable)
    for i in range(n - 1):
        x = residual_block(x, filters, conv_start_idx+2*(1+i), is_test, trainable)
    return x

def YOLOv3(inputs, initial_filters, num_classes, is_test, trainable=True):
    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32

    ''' darknet53部分，所有卷积层都没有偏移bias_attr=False '''
    x = conv2d_unit(inputs, i32, (3, 3), stride=1, padding=1, name='conv01', is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, i64, (3, 3), stride=2, padding=1, name='conv02', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i32, n=1, conv_start_idx=3, is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, i128, (3, 3), stride=2, padding=1, name='conv05', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i64, n=2, conv_start_idx=6, is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, i256, (3, 3), stride=2, padding=1, name='conv10', is_test=is_test, trainable=trainable)
    act11 = stack_residual_block(x, i128, n=8, conv_start_idx=11, is_test=is_test, trainable=trainable)

    x = conv2d_unit(act11, i512, (3, 3), stride=2, padding=1, name='conv27', is_test=is_test, trainable=trainable)
    act19 = stack_residual_block(x, i256, n=8, conv_start_idx=28, is_test=is_test, trainable=trainable)

    x = conv2d_unit(act19, i1024, (3, 3), stride=2, padding=1, name='conv44', is_test=is_test, trainable=trainable)
    act23 = stack_residual_block(x, i512, n=4, conv_start_idx=45, is_test=is_test, trainable=trainable)
    ''' darknet53部分结束，余下部分不再有残差块stack_residual_block() '''

    x = conv2d_unit(act23, i512, (1, 1), stride=1, padding=0, name='conv53', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, (3, 3), stride=1, padding=1, name='conv54', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, (1, 1), stride=1, padding=0, name='conv55', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, (3, 3), stride=1, padding=1, name='conv56', is_test=is_test, trainable=trainable)
    lkrelu57 = conv2d_unit(x, i512, (1, 1), stride=1, padding=0, name='conv57', is_test=is_test, trainable=trainable)

    x = conv2d_unit(lkrelu57, i1024, (3, 3), stride=1, padding=1, name='conv58', is_test=is_test, trainable=trainable)
    y1 = P.conv2d(x, 3 * (num_classes + 5), filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="conv59.conv.weights"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="conv59.conv.bias"))

    x = conv2d_unit(lkrelu57, i256, (1, 1), stride=1, padding=0, name='conv60', is_test=is_test, trainable=trainable)
    x = P.resize_nearest(x, scale=float(2))
    x = P.concat([x, act19], axis=1)

    x = conv2d_unit(x, i256, (1, 1), stride=1, padding=0, name='conv61', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, (3, 3), stride=1, padding=1, name='conv62', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, (1, 1), stride=1, padding=0, name='conv63', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, (3, 3), stride=1, padding=1, name='conv64', is_test=is_test, trainable=trainable)
    lkrelu64 = conv2d_unit(x, i256, (1, 1), stride=1, padding=0, name='conv65', is_test=is_test, trainable=trainable)

    x = conv2d_unit(lkrelu64, i512, (3, 3), stride=1, padding=1, name='conv66', is_test=is_test, trainable=trainable)
    y2 = P.conv2d(x, 3 * (num_classes + 5), filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="conv67.conv.weights"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="conv67.conv.bias"))

    x = conv2d_unit(lkrelu64, i128, (1, 1), stride=1, padding=0, name='conv68', is_test=is_test, trainable=trainable)
    x = P.resize_nearest(x, scale=float(2))
    x = P.concat([x, act11], axis=1)

    x = conv2d_unit(x, i128, (1, 1), stride=1, padding=0, name='conv69', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, (3, 3), stride=1, padding=1, name='conv70', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, (1, 1), stride=1, padding=0, name='conv71', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, (3, 3), stride=1, padding=1, name='conv72', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, (1, 1), stride=1, padding=0, name='conv73', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, (3, 3), stride=1, padding=1, name='conv74', is_test=is_test, trainable=trainable)
    y3 = P.conv2d(x, 3 * (num_classes + 5), filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="conv75.conv.weights"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="conv75.conv.bias"))

    # 相当于numpy的transpose()，交换下标
    y1 = P.transpose(y1, perm=[0, 2, 3, 1])
    y2 = P.transpose(y2, perm=[0, 2, 3, 1])
    y3 = P.transpose(y3, perm=[0, 2, 3, 1])
    return y1, y2, y3

