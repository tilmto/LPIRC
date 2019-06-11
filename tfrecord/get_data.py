import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_for_eval(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=True):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
    central_crop: Enable central cropping of images during preprocessing for
      evaluation.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_crop and central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def read_and_decode(filename_list):
    filename_queue = tf.train.string_input_producer(filename_list)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                           'image/height' : tf.FixedLenFeature([], tf.int64),
                                           'image/width' : tf.FixedLenFeature([], tf.int64),
                                       })

    label = tf.cast(features['image/class/label'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    img = tf.image.decode_jpeg(features['image/encoded'],channels=3)
    img = preprocess_for_eval(img,224,224)
    #img = tf.reshape(img, [height, width, 3])
    #img = img/255 - 0.5

    img_batch, label_batch = tf.train.shuffle_batch([img,label],batch_size=1,num_threads=2,capacity=100,min_after_dequeue=10)

    return img_batch,label_batch

if __name__ == '__main__':
    filename_list = ['validation-00046-of-00128']
    img_batch,label_batch = read_and_decode(filename_list)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(20):
            image,label = sess.run([img_batch,label_batch])
            print(image.shape,label)

            plt.imshow(image[0])
            plt.show()

        coord.request_stop()
        coord.join(threads)

