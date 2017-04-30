import tensorflow as tf
from blocks import build_blocks

class Generator:
    """Generator from CycleGAN Paper"""
    def __init__(self, name, is_training=True, verbose=False):
        self.name = name
        self.is_training = is_training
        self.variables = None

    def __call__(self, inputs, params):
        with tf.variable_scope(self.name):
            output = build_blocks([
                'c7s1-32', 'd64', 'd128', 'R128x6', 'u64', 'u32', 'c7s1-3'
            ], params, is_training=self.is_training, verbose=self.verbose)(inputs)
            output = output = tf.nn.tanh(output, name='final-tanh')

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
