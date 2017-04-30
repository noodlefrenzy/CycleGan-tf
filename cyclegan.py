import tensorflow as tf
from discriminator import Discriminator
from generator import Generator
from images import batch_to_image

class CycleGAN:
    def __init__(self, batch_size=1, image_size=256, start_lr=0.0002, lambdas=(10., 10.), lsgan=True, betas=(0.5, 0.9), verbose=False):
        self.lambdas = lambdas
        self.use_least_square_loss = lsgan
        self.betas = betas
        self.starting_learning_rate = start_lr
        self.verbose = verbose
        self.default_params = {
            'data_format': 'channels_first',
            'padding': 'same',
            'epsilon': 1e-5,
            'momentum': 0.8,
            'reuse': False
        }

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator('G', self.default_params, is_training=self.is_training, verbose=self.verbose)
        self.D_Y = Discriminator('D_Y', self.default_params, is_training=self.is_training, verbose=self.verbose)
        self.F = Generator('F', self.default_params, is_training=self.is_training, verbose=self.verbose)
        self.D_X = Discriminator('D_X', self.default_params, is_training=self.is_training, verbose=self.verbose)

        self.fake_x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])

    def model(self, x, y):
        cyc_loss = self.cycle_loss(self.G, self.F, x, y)

        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y)
        G_loss =  G_gan_loss + cyc_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y)

        fake_x = self.F(y)
        F_gan_loss = self.generator_loss(self.D_X, fake_x)
        F_loss = F_gan_loss + cyc_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x)

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cyc_loss)

        tf.summary.image('X/generated', batch_to_image(self.G(x)))
        tf.summary.image('X/reconstruction', batch_to_image(self.F(self.G(x))))
        tf.summary.image('Y/generated', batch_to_image(self.F(y)))
        tf.summary.image('Y/reconstruction', batch_to_image(self.G(self.F(y))))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def adam(loss, variables, name_prefix):
            name = name_prefix + '_adam'
            global_step = tf.Variable(0, trainable=False)
            # The paper recommends learning at a fixed rate for several steps, and then linearly stepping down to 0
            learning_rate = (tf.where(tf.greater_equal(global_step, 100000),
                 tf.train.polynomial_decay(self.starting_learning_rate, global_step - 100000, 100000, 0.0, power=1.0),
                 self.starting_learning_rate))
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (tf.train.AdamOptimizer(learning_rate, beta1=self.betas[0], name=name).minimize(
                loss, global_step=global_step, var_list=variables))
            return learning_step

        G_optimizer = adam(G_loss, self.G.variables, 'G')
        D_Y_optimizer = adam(D_Y_loss, self.D_Y.variables, 'D_Y')
        F_optimizer =  adam(F_loss, self.F.variables, 'F')
        D_X_optimizer = adam(D_X_loss, self.D_X.variables, 'D_X')

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self, D, y, fake_y):
        if self.use_least_square_loss:
            # use least square (as in paper)
            error_real = tf.reduce_mean(tf.squared_difference(D(y), 1.))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use negative log-likelihood (+ e to avoid errors at zero)
            error_real = -tf.reduce_mean(tf.log(D(y) + 1e-12))
            error_fake = -tf.reduce_mean(tf.log(1 - D(fake_y) + 1e-12))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y):
        if self.use_least_square_loss:
            # use least square (as in paper)
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), 1.))
        else:
            # use negative log-likelihood (+ e to avoid errors at zero)
            loss = -tf.reduce_mean(tf.log(D(fake_y) + 1e-12)) / 2
        return loss

    def cycle_loss(self, F, G, x, y):
        forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
        loss = self.lambdas[0] * forward_loss + self.lambdas[1] * backward_loss
        return loss
