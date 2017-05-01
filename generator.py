import tensorflow as tf
from blocks import build_blocks

class Generator:
    """Generator from CycleGAN Paper"""
    def __init__(self, name, params, is_training=True, verbose=False):
        self.name = name
        self.params = params.copy()
        self.is_training = is_training
        self.variables = None
        self.verbose = verbose

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.params['reuse']):
            if self.verbose:
                print('Building Generator %s' % self.name)
            output = build_blocks([
                'c7s1-32', 'd64', 'd128', 'R128x9', 'u64', 'u32', 'c7s1-3-T'
            ], self.params, is_training=self.is_training, verbose=self.verbose)(inputs)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.params['reuse'] = True

        return output