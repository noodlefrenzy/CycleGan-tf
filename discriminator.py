import tensorflow as tf
from blocks import build_blocks

class Discriminator:
    """Discriminator from CycleGAN Paper"""
    def __init__(self, name, params, is_training=True, verbose=False):
        self.name = name
        self.params = params.copy()
        self.is_training = is_training
        self.variables = None
        self.verbose = verbose

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.params['reuse']):
            if self.verbose:
                print('Building Discriminator %s' % self.name)
            output = build_blocks([
                'C64N', 'C128', 'C256', 'C512x2',
            ], self.params, is_training=self.is_training, verbose=self.verbose)(inputs)
            output = tf.layers.conv2d(output, filters=1, kernel_size=1, padding=self.params['padding'],
              data_format=self.params['data_format'], name='final_reshape')

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.params['reuse'] = True

        return output