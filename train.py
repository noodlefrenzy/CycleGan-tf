from cyclegan import CycleGAN
from images import Images, ImageCache
from datetime import datetime
import logging
import os
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='Train a CycleGAN.')
parser.add_argument('--bs', '--batch-size', type=int, default=1, help='Batch size', dest='batch_size')
parser.add_argument('--im', '--image-size', type=int, default=256, help='Image size (images are square)', dest='image_size')
parser.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of epochs to train', dest='num_epochs')
parser.add_argument('-t', '--num-threads', type=int, default=2, help='Number of threads', dest='num_threads')
parser.add_argument('--lsl', '--least-square-loss', action='store_true',
                    help='Whether to use least-square loss (defaults to log-likelihood)', dest='use_least_square')
parser.add_argument('--l1', '--lambda1', type=float, default=10., help='Weight for cycle loss (forward)', dest='lambda_1')
parser.add_argument('--l2', '--lambda2', type=float, default=10., help='Weight for cycle loss (reverse)', dest='lambda_2')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.0002, help='Starting learning rate', dest='learning_rate')
parser.add_argument('--b1', '--beta1', type=float, default=0.5, help='Momentum for Adam optimizer (G)', dest='beta_1')
parser.add_argument('--b2', '--beta2', type=float, default=0.5, help='Momentum for Adam optimizer (D)', dest='beta_2')
parser.add_argument('-i', '--input-prefix', help='Prefix path name to tfrecords files.', required=True, dest='input_prefix')
parser.add_argument('-c', '--checkpoint-dir', default='./checkpoints', help='Checkpoint directory', dest='checkpoint_dir')

def train(args):
    now = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = os.path.join(args.checkpoint_dir, now)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logging.info('Checkpointing to "{}"'.format(checkpoints_dir))

    infile_X, infile_Y = ['{}_{}.tfrecords'.format(args.input_prefix, p) for p in ['trainA', 'trainB']]
    logging.info('Loading data from "{}" and "{}"'.format(infile_X, infile_Y))

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size=args.batch_size,
            image_size=args.image_size,
            lsgan=args.use_least_square,
            lambdas=(args.lambda_1, args.lambda_2),
            start_lr=args.learning_rate,
            betas=(args.beta_1, args.beta_2),
            verbose=False
        )
        inputs_X = Images(infile_X, batch_size=args.batch_size, image_size=args.image_size, num_threads=args.num_threads, name='X')
        inputs_Y = Images(infile_Y, batch_size=args.batch_size, image_size=args.image_size, num_threads=args.num_threads, name='Y')

        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model(inputs_X.feed(), inputs_Y.feed())
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        try:
            fake_X_cache = ImageCache()
            fake_Y_cache = ImageCache()
            while not coord.should_stop():
                cur_fake_y, cur_fake_x = sess.run([fake_y, fake_x])

                _, cur_G_loss, cur_D_Y_loss, cur_F_loss, cur_D_X_loss, summary = (
                    sess.run(
                        [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                        feed_dict={cycle_gan.fake_y: fake_Y_cache.fetch(cur_fake_y),
                                   cycle_gan.fake_x: fake_X_cache.fetch(cur_fake_x)}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 100 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  G_loss   : {}'.format(cur_G_loss))
                    logging.info('  D_Y_loss : {}'.format(cur_D_Y_loss))
                    logging.info('  F_loss   : {}'.format(cur_F_loss))
                    logging.info('  D_X_loss : {}'.format(cur_D_X_loss))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(parsed_args):
    train(parsed_args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = parser.parse_args()
    main(result)
