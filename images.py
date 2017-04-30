import tensorflow as tf

class Images:
    def __init__(self, path, num_epochs, batch_size=1, num_threads=2, image_size=256, crop_size=224, \
                 should_crop=True, should_flip=True. should_shuffle=True):
        self.path = path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.image_size = image_size
        self.crop_size = crop_size
        self.should_crop = should_crop
        self.should_flip = should_flip
        self.should_shuffle = should_shuffle

    def feed(self):
        filenames = tf.train.match_filenames_once(self.path)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=self.num_epochs, \
            shuffle=self.should_shuffle)
        reader = tf.WholeFileReader()
        filename, value = reader.read(filename_queue)

        image = tf.image.decode_jpeg(value, channels=3)

        processed = tf.image.resize_images(
            image,
            [self.image_size, self.image_size],
            tf.image.ResizeMethod.BILINEAR)

        if self.should_flip:
            processed = tf.image.random_flip_left_right(processed)
        if self.should_crop:
            processed = tf.random_crop(processed, [self.crop_size, self.crop_size, 3])
        # CHANGE TO 'CHW' DATA_FORMAT FOR FASTER GPU PROCESSING
        processed = tf.transpose(processed, [2, 0, 1])
        processed = (tf.cast(processed, tf.float32) - 128.0) / 128.0

        images = tf.train.batch(
            [processed],
            batch_size = self.batch_size,
            num_threads = self.num_threads,
            capacity = self.batch_size * 5)

        return images
