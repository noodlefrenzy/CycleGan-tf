import logging
import os
import tensorflow as tf

import argparse

import random

random.seed(1337)  # Set seed for repeatable shuffling.

def exists(x):
    for d in ['', 'trainA', 'trainB', 'testA', 'testB']:
        path = os.path.join(x, d)
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(
                '"%s" must be a valid directory containing trainA, trainB, testA, and testB dirs.' % x)
    return x

parser = argparse.ArgumentParser(description='Convert training and testing images to tfrecords files.')
parser.add_argument('--files', type=exists, help='Location of training files', required=True)
parser.add_argument('-o', '--output-prefix', help='Prefix file/path for output tfrecords files', required=True,
                    dest='output_prefix')

def reader(path, shuffle=True):
    files = []

    for img_file in os.scandir(path):
        if img_file.name.endswith('.jpg') and img_file.is_file():
            files.append(img_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(files)))
        random.shuffle(shuffled_index)

        files = [files[i] for i in shuffled_index]

    return files

def writer(root_path, output_prefix):
    as_bytes = lambda data: tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
    as_example = lambda fn, data: tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': as_bytes(tf.compat.as_bytes(os.path.basename(fn))),
        'image/encoded_image': as_bytes((data))
    }))

    for sub in ['trainA', 'trainB', 'testA', 'testB']:
        indir = os.path.join(root_path, sub)
        outfile = os.path.abspath('{}_{}.tfrecords'.format(output_prefix, sub))
        files = reader(indir)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        record_writer = tf.python_io.TFRecordWriter(outfile)

        for ix, file in enumerate(files):
            with tf.gfile.FastGFile(file, 'rb') as f:
                image_data = f.read()

            example = as_example(file, image_data)
            record_writer.write(example.SerializeToString())

            if ix % 500 == 0:
                print('Processed {}/{}.'.format(ix, len(files)))
        print('Done.')
        record_writer.close()

def main(parsed_args):
    writer(parsed_args.files, parsed_args.output_prefix)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = parser.parse_args()
    main(result)
