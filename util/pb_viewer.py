import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session(graph=tf.Graph()) as sess:
	tf.saved_model.loader.load(sess, ["serve"], "/home/luke/mobilenet/saved_model")
	graph = tf.get_default_graph()
	LOGDIR='output'
	train_writer = tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)
	train_writer.flush()
	train_writer.close()
#with tf.Graph().as_default() as graph:
#	tf.import_graph_def('/home/luke/mnasnet/mnasnet-a1/saved_model/saved_model.pb')
#	with tf.Session() as sess:
#		LOGDIR='output'
#		train_writer = tf.summary.FileWriter(LOGDIR)
#		train_writer.add_graph(sess.graph)
#		train_writer.flush()
#		train_writer.close()
#with tf.Session() as sess:
#	 model_filename ='/home/luke/mnasnet/mnasnet-a1/saved_model/saved_model.pb'
#	 with tf.gfile.GFile(model_filename, 'rb') as f:
#		 graph_def = tf.GraphDef()
#		 graph_def.ParseFromString(f.read())
#		 g_in = tf.import_graph_def(graph_def)
#LOGDIR='output'
#train_writer = tf.summary.FileWriter(LOGDIR)
#train_writer.add_graph(sess.graph)
#train_writer.flush()
#train_writer.close()
