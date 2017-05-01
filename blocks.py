#
# Layers
#
# Simple library for building layers in the format outlined in the CycleGAN paper.
#
import re
import tensorflow as tf

#
# From the paper:
#  Let c7s1-k denote a 7 × 7 Convolution-BatchNormReLU layer with k ﬁlters and stride 1.
#  dk denotes a 3 × 3 Convolution-BatchNorm-ReLU layer with k ﬁlters, and stride 2.
#   Reﬂection padding was used to reduce artifacts.
#  Rk denotes a residual block that contains two 3×3 convolutional layers
#   with the same number of ﬁlters on both layer.
#  uk denotes a 3 × 3 fractional-strided-ConvolutionBatchNorm-ReLU layer with k ﬁlters, and stride 1/2.
#
# Let's build regex to match these and pull out the components:
block_pattern = re.compile('''
((
c                           # Conv/BatchNorm/Relu
(?P<kernel_size>\d+)        # Kernel size
s(?P<stride>\d+)            # Stride
-
)|(?P<block_type>[dRuC]))   # The other blocks have a consistent pattern
(?P<num_filters>\d+)        # Number of filters
(?P<norm_type>[BIN])?       # Batch, Instance, or No normalization 
-?(?P<activation_fn>[LT])?  # Type of activation function (if not ReLU)
x?(?P<repeats>\d+)?         # Number of times to repeat
''', re.VERBOSE)

def leaky_relu(inputs, alpha=0.2, name='leaky_relu'):
    """Activation function for 'leaky' ReLU"""
    return tf.maximum(alpha * inputs, inputs, name=name)

def norm(inputs, params, is_training=True):
    if params['norm_type'] == 'B':
        with tf.variable_scope('batch_norm'):
            return tf.layers.batch_normalization(inputs, training=is_training)
    elif params['norm_type'] == 'I':
        with tf.variable_scope('instance_norm'):
            depth = inputs.get_shape()[3]
            scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02))
            offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
            inv = tf.rsqrt(variance + 1e-5)
            normalized = (inputs - mean) * inv
            return scale * normalized + offset
    else:
        return inputs

def conv_block(params):
    """Creates Convolution Block Builder from given parameters"""
    def builder(inputs, is_training=True, verbose=False):
        """Convolution Block Builder"""
        if verbose:
            print('  {}: Creating convolution{} block w/ kernel size {}, stride {}, and {} filters.{}'.\
                format(params['name'], ' (Transpose)' if params['is_transpose'] else '',
                    params['kernel_size'], params['stride'], params['num_filters'],
                    ' No Batch Normalization.' if params['skip_batch_norm'] else ''))
        with tf.variable_scope(params['name'], reuse=params['reuse']):
            conv = tf.layers.conv2d_transpose if params['is_transpose'] else tf.layers.conv2d
            output = conv(inputs, filters=params['num_filters'], kernel_size=params['kernel_size'],
                strides=params['stride'], padding=params['padding'],
                data_format=params['data_format'], name='prebatch')
            output = norm(output, params, is_training=is_training)
            output = params['activation_fn'](output, name='activation')
            return output
    return builder

def constrained_conv_batchnorm(kernel_size, stride, activation_fn, params):
    """Convolution Block builder with constraints"""
    params['kernel_size'] = kernel_size
    params['stride'] = stride
    if activation_fn is not None:
        params['activation_fn'] = activation_fn
    return conv_block(params)

def residual_block(params):
    """Creates Residual Block Builder from given parameters"""
    if 'kernel_size' not in params:
        params['kernel_size'] = 3
    if 'stride' not in params:
        params['stride'] = 1
    def builder(inputs, is_training=True, verbose=False):
        """Residual Block Builder"""
        if verbose:
            print('  {}: Creating residual block w/ kernel size {}, stride {}, and {} filters'.format(
                params['name'], params['kernel_size'], params['stride'], params['num_filters']))
        with tf.variable_scope(params['name'], reuse=params['reuse']):
            conv = tf.layers.conv2d_transpose if params['is_transpose'] else tf.layers.conv2d
            output = conv(inputs, filters=params['num_filters'], kernel_size=params['kernel_size'],
                strides=params['stride'], padding=params['padding'],
                data_format=params['data_format'], name='conv2d-1')
            output = params['activation_fn'](output, name='activation')
            output = conv(output, filters=params['num_filters'], kernel_size=params['kernel_size'],
                strides=params['stride'], padding=params['padding'],
                data_format=params['data_format'], name='conv2d-2')
            return inputs + output
    return builder

def fractional_conv_batchnorm(params):
    """Deconvolution (Conv-Transpose) Builder"""
    params['kernel_size'] = 3
    params['stride'] = 2
    params['is_transpose'] = True
    return conv_block(params)

activation_functions = {
    'L': leaky_relu,
    'T': tf.nn.tanh,
    'relu': tf.nn.relu
}

block_builders = {
    'c': conv_block,
    'd': lambda params: constrained_conv_batchnorm(3, 2, None, params),
    'R': residual_block,
    'u': fractional_conv_batchnorm,
    'C': lambda params: constrained_conv_batchnorm(4, 2, leaky_relu, params)
}

def cvt_dict(matchdict, default_params):
    """Converts a 'Match' group dictionary to block parameters"""
    params = default_params.copy()
    params['is_transpose'] = False
    params['repeats'] = 1
    for k in ['kernel_size', 'stride', 'num_filters', 'repeats']:
        if matchdict[k] is not None:
            params[k] = int(matchdict[k])
    params['activation_fn'] = activation_functions[matchdict['activation_fn'] or 'relu']
    params['block_type'] = matchdict['block_type'] or 'c'
    params['norm_type'] = matchdict['norm_type'] or 'I'
    return params

def build_blocks(block_list, default_params, is_training=True, verbose=False):
    """Build a collection of chained blocks from their string descriptions"""
    def builder(inputs):
        """Block collection builder"""
        output = inputs
        names = {}
        for block_str in block_list:
            match = block_pattern.fullmatch(block_str)
            if match is None:
                raise Exception('Unable to match block type from ' + block_str)
            cur_params = cvt_dict(match.groupdict(), default_params)
            builder = block_builders[cur_params['block_type']]
            for _ in range(cur_params['repeats']):
                cur_name = block_str
                if cur_name in names:
                    names[cur_name] += 1
                    cur_name = block_str + '-' + str(names[block_str])
                else:
                    names[cur_name] = 1
                cur_params['name'] = cur_name
                output = builder(cur_params)(output, is_training=is_training, verbose=verbose)
        return output
    return builder
