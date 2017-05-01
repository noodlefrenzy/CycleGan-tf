import tensorflow as tf
import random

def to_image(data):
    return tf.image.convert_image_dtype((data + 1.) / 2., tf.uint8)

def batch_to_image(batch):
    return tf.map_fn(to_image, batch, dtype=tf.uint8)

class Images():
  def __init__(self, tfrecords_file, image_size=256, batch_size=1, num_threads=2, name=''):
    self.tfrecords_file = tfrecords_file
    self.image_size = image_size
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.name = name

  def feed(self):
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      reader = tf.TFRecordReader()

      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
          })

      image_buffer = features['image/encoded_image']
      image = tf.image.decode_jpeg(image_buffer, channels=3)
      image = self.preprocess(image)
      images = tf.train.shuffle_batch(
            [image], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=100 + 3*self.batch_size,
            min_after_dequeue=100
          )

      tf.summary.image('_input', images)
    return images

  def preprocess(self, image):
    image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = (image / 127.5) - 1.
    image.set_shape([self.image_size, self.image_size, 3])
    return image


class ImageCache:
    def __init__(self, cache_size=30):
        self.cache_size = cache_size
        self.images = []

    def fetch(self, image):
        if self.cache_size == 0:
            return image

        p = random.random()
        if p > 0.5 and len(self.images) > 0:
            # use and replace old image.
            random_id = random.randrange(len(self.images))
            retval = self.images[random_id].copy()
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            else:
                self.images[random_id] = image.copy()
            return retval
        else:
            if len(self.images) < self.cache_size:
                self.images.append(image.copy())
            return image
