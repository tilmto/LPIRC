from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

with tf.Session() as sess:
    checkpoint_path="model.ckpt-1"
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_shape_map.pop('global_step')
    for key in var_to_shape_map:
        val = reader.get_tensor(key)
        tf.Variable(val,name=key)

    tf.Variable(0,dtype=tf.int64,name='global_step')

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess,'model.ckpt-0')



