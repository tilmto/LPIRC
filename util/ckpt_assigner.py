import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

global_step=0
n0=0
with tf.Session() as sess:
	checkpoint_path = os.path.join("./dst_ckpt/", "model.ckpt-0") #dest
	checkpoint_path2=os.path.join("./efficient_net_ckpt", "model.ckpt") #src can be a pytorch reader which may invole some modification
	reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
	reader2 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path2)
	var_to_shape_map = reader.get_variable_to_shape_map()
	var_to_shape_map2 = reader2.get_variable_to_shape_map()
	n=16
	#iterate over expanded convs
	for i in range(0,n):
		for key in var_to_shape_map:
			#var = tf.Variable(reader.get_tensor(key), name=key)
			if i==0:
				tmp="/expanded_conv/"
			else:
				tmp="/expanded_conv_"+str(i)+"/"
			if tmp in key:
				overhead="efficientnet-b0/blocks_"+str(i)+"/"
				if i==0:
					#depthwise
					if "/depthwise/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/moving_variance/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/moving_variance" in key:
						vname=overhead+"tpu_batch_normalization/moving_variance"
					elif "/depthwise/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/moving_mean/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/moving_mean" in key:
						vname=overhead+"tpu_batch_normalization/moving_mean"

					elif "depthwise/BatchNorm/gamma/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/gamma/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/gamma/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization/gamma/RMSProp_1"
					elif "depthwise/BatchNorm/gamma/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization/gamma/RMSProp"
					elif "depthwise/BatchNorm/gamma" in key:
						vname=overhead+"tpu_batch_normalization/gamma"

					elif "depthwise/BatchNorm/beta/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/beta/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/beta/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization/beta/RMSProp_1"
					elif "depthwise/BatchNorm/beta/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization/beta/RMSProp"
					elif "depthwise/BatchNorm/beta" in key:
						vname=overhead+"tpu_batch_normalization/beta"

					elif "depthwise/depthwise_weights/RMSProp_1" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/RMSProp_1"
					elif "depthwise/depthwise_weights/RMSProp" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/RMSProp"
					elif "depthwise/depthwise_weights/ExponentialMovingAverage" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage"
					elif "depthwise/depthwise_weights" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel"
					#se
					elif "se_1/biases/RMSProp_1" in key:
						vname=overhead+"se/conv2d/bias/RMSProp_1"
					elif "se_1/biases/RMSProp" in key:
						vname=overhead+"se/conv2d/bias/RMSProp"
					elif "se_1/biases/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d/bias/ExponentialMovingAverage"
					elif "se_1/biases" in key:
						vname=overhead+"se/conv2d/bias"

					elif "se_1/weights/RMSProp_1" in key:
						vname=overhead+"se/conv2d/kernel/RMSProp_1"
					elif "se_1/weights/RMSProp" in key:
						vname=overhead+"se/conv2d/kernel/RMSProp"
					elif "se_1/weights/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d/kernel/ExponentialMovingAverage"
					elif "se_1/weights" in key:
						vname=overhead+"se/conv2d/kernel"

					elif "se_2/biases/RMSProp_1" in key:
						vname=overhead+"se/conv2d_1/bias/RMSProp_1"
					elif "se_2/biases/RMSProp" in key:
						vname=overhead+"se/conv2d_1/bias/RMSProp"
					elif "se_2/biases/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d_1/bias/ExponentialMovingAverage"
					elif "se_2/biases" in key:
						vname=overhead+"se/conv2d_1/bias"

					elif "se_2/weights/RMSProp_1" in key:
						vname=overhead+"se/conv2d_1/kernel/RMSProp_1"
					elif "se_2/weights/RMSProp" in key:
						vname=overhead+"se/conv2d_1/kernel/RMSProp"
					elif "se_2/weights/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d_1/kernel/ExponentialMovingAverage"
					elif "se_2/weights" in key:
						vname=overhead+"se/conv2d_1/kernel"

					#project
					elif "/project/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage"
					elif "project/BatchNorm/moving_variance" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_variance"
					elif "/project/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage"
					elif "project/BatchNorm/moving_mean" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_mean"

					elif "project/BatchNorm/gamma/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/ExponentialMovingAverage"
					elif "project/BatchNorm/gamma/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/RMSProp_1"
					elif "project/BatchNorm/gamma/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/RMSProp"
					elif "project/BatchNorm/gamma" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma"

					elif "project/BatchNorm/beta/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/ExponentialMovingAverage"
					elif "project/BatchNorm/beta/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/RMSProp_1"
					elif "project/BatchNorm/beta/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/RMSProp"
					elif "project/BatchNorm/beta" in key:
						vname=overhead+"tpu_batch_normalization_1/beta"

					elif "project/weights/RMSProp_1" in key:
						vname=overhead+"conv2d/kernel/RMSProp_1"
					elif "project/weights/RMSProp" in key:
						vname=overhead+"conv2d/kernel/RMSProp"
					elif "project/weights/ExponentialMovingAverage" in key:
						vname=overhead+"conv2d/kernel/ExponentialMovingAverage"
					elif "project/weights" in key:
						vname=overhead+"conv2d/kernel"
				else:
					#se
					if "se_1/biases/RMSProp_1" in key:
						vname=overhead+"se/conv2d/bias/RMSProp_1"
					elif "se_1/biases/RMSProp" in key:
						vname=overhead+"se/conv2d/bias/RMSProp"
					elif "se_1/biases/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d/bias/ExponentialMovingAverage"
					elif "se_1/biases" in key:
						vname=overhead+"se/conv2d/bias"

					elif "se_1/weights/RMSProp_1" in key:
						vname=overhead+"se/conv2d/kernel/RMSProp_1"
					elif "se_1/weights/RMSProp" in key:
						vname=overhead+"se/conv2d/kernel/RMSProp"
					elif "se_1/weights/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d/kernel/ExponentialMovingAverage"
					elif "se_1/weights" in key:
						vname=overhead+"se/conv2d/kernel"

					elif "se_2/biases/RMSProp_1" in key:
						vname=overhead+"se/conv2d_1/bias/RMSProp_1"
					elif "se_2/biases/RMSProp" in key:
						vname=overhead+"se/conv2d_1/bias/RMSProp"
					elif "se_2/biases/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d_1/bias/ExponentialMovingAverage"
					elif "se_2/biases" in key:
						vname=overhead+"se/conv2d_1/bias"

					elif "se_2/weights/RMSProp_1" in key:
						vname=overhead+"se/conv2d_1/kernel/RMSProp_1"
					elif "se_2/weights/RMSProp" in key:
						vname=overhead+"se/conv2d_1/kernel/RMSProp"
					elif "se_2/weights/ExponentialMovingAverage" in key:
						vname=overhead+"se/conv2d_1/kernel/ExponentialMovingAverage"
					elif "se_2/weights" in key:
						vname=overhead+"se/conv2d_1/kernel"
						
					#project
					elif "/project/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_2/moving_variance/ExponentialMovingAverage"
					elif "project/BatchNorm/moving_variance" in key:
						vname=overhead+"tpu_batch_normalization_2/moving_variance"
					elif "/project/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_2/moving_mean/ExponentialMovingAverage"
					elif "project/BatchNorm/moving_mean" in key:
						vname=overhead+"tpu_batch_normalization_2/moving_mean"

					elif "project/BatchNorm/gamma/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_2/gamma/ExponentialMovingAverage"
					elif "project/BatchNorm/gamma/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_2/gamma/RMSProp_1"
					elif "project/BatchNorm/gamma/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_2/gamma/RMSProp"
					elif "project/BatchNorm/gamma" in key:
						vname=overhead+"tpu_batch_normalization_2/gamma"

					elif "project/BatchNorm/beta/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_2/beta/ExponentialMovingAverage"
					elif "project/BatchNorm/beta/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_2/beta/RMSProp_1"
					elif "project/BatchNorm/beta/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_2/beta/RMSProp"
					elif "project/BatchNorm/beta" in key:
						vname=overhead+"tpu_batch_normalization_2/beta"

					elif "project/weights/RMSProp_1" in key:
						vname=overhead+"conv2d_1/kernel/RMSProp_1"
					elif "project/weights/RMSProp" in key:
						vname=overhead+"conv2d_1/kernel/RMSProp"
					elif "project/weights/ExponentialMovingAverage" in key:
						vname=overhead+"conv2d_1/kernel/ExponentialMovingAverage"
					elif "project/weights" in key:
						vname=overhead+"conv2d_1/kernel"

					#depthwise
					elif "/depthwise/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/moving_variance" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_variance"
					elif "/depthwise/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/moving_mean" in key:
						vname=overhead+"tpu_batch_normalization_1/moving_mean"

					elif "depthwise/BatchNorm/gamma/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/gamma/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/RMSProp_1"
					elif "depthwise/BatchNorm/gamma/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma/RMSProp"
					elif "depthwise/BatchNorm/gamma" in key:
						vname=overhead+"tpu_batch_normalization_1/gamma"

					elif "depthwise/BatchNorm/beta/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/ExponentialMovingAverage"
					elif "depthwise/BatchNorm/beta/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/RMSProp_1"
					elif "depthwise/BatchNorm/beta/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization_1/beta/RMSProp"
					elif "depthwise/BatchNorm/beta" in key:
						vname=overhead+"tpu_batch_normalization_1/beta"

					elif "depthwise/depthwise_weights/RMSProp_1" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/RMSProp_1"
					elif "depthwise/depthwise_weights/RMSProp" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/RMSProp"
					elif "depthwise/depthwise_weights/ExponentialMovingAverage" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage"
					elif "depthwise/depthwise_weights" in key:
						vname=overhead+"depthwise_conv2d/depthwise_kernel"
						
					#expand
					elif "/expand/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/moving_variance/ExponentialMovingAverage"
					elif "expand/BatchNorm/moving_variance" in key:
						vname=overhead+"tpu_batch_normalization/moving_variance"
					elif "/expand/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/moving_mean/ExponentialMovingAverage"
					elif "expand/BatchNorm/moving_mean" in key:
						vname=overhead+"tpu_batch_normalization/moving_mean"

					elif "expand/BatchNorm/gamma/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/gamma/ExponentialMovingAverage"
					elif "expand/BatchNorm/gamma/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization/gamma/RMSProp_1"
					elif "expand/BatchNorm/gamma/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization/gamma/RMSProp"
					elif "expand/BatchNorm/gamma" in key:
						vname=overhead+"tpu_batch_normalization/gamma"

					elif "expand/BatchNorm/beta/ExponentialMovingAverage" in key:
						vname=overhead+"tpu_batch_normalization/beta/ExponentialMovingAverage"
					elif "expand/BatchNorm/beta/RMSProp_1" in key:
						vname=overhead+"tpu_batch_normalization/beta/RMSProp_1"
					elif "expand/BatchNorm/beta/RMSProp" in key:
						vname=overhead+"tpu_batch_normalization/beta/RMSProp"
					elif "expand/BatchNorm/beta" in key:
						vname=overhead+"tpu_batch_normalization/beta"

					elif "expand/weights/RMSProp_1" in key:
						vname=overhead+"conv2d/kernel/RMSProp_1"
					elif "expand/weights/RMSProp" in key:
						vname=overhead+"conv2d/kernel/RMSProp"
					elif "expand/weights/ExponentialMovingAverage" in key:
						vname=overhead+"conv2d/kernel/ExponentialMovingAverage"
					elif "expand/weights" in key:
						vname=overhead+"conv2d/kernel"
				new_val=reader2.get_tensor(vname)
				var = tf.Variable(new_val, name=key)
				n0=n0+1

	#handle heading and ending
	for key in var_to_shape_map:
		if "/Conv/" in key:
			overhead="efficientnet-b0/stem/"
			#Conv
			if "/Conv/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/moving_variance/ExponentialMovingAverage"
			elif "Conv/BatchNorm/moving_variance" in key:
				vname=overhead+"tpu_batch_normalization/moving_variance"
			elif "/Conv/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/moving_mean/ExponentialMovingAverage"
			elif "Conv/BatchNorm/moving_mean" in key:
				vname=overhead+"tpu_batch_normalization/moving_mean"

			elif "Conv/BatchNorm/gamma/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/gamma/ExponentialMovingAverage"
			elif "Conv/BatchNorm/gamma/RMSProp_1" in key:
				vname=overhead+"tpu_batch_normalization/gamma/RMSProp_1"
			elif "Conv/BatchNorm/gamma/RMSProp" in key:
				vname=overhead+"tpu_batch_normalization/gamma/RMSProp"
			elif "Conv/BatchNorm/gamma" in key:
				vname=overhead+"tpu_batch_normalization/gamma"

			elif "Conv/BatchNorm/beta/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/beta/ExponentialMovingAverage"
			elif "Conv/BatchNorm/beta/RMSProp_1" in key:
				vname=overhead+"tpu_batch_normalization/beta/RMSProp_1"
			elif "Conv/BatchNorm/beta/RMSProp" in key:
				vname=overhead+"tpu_batch_normalization/beta/RMSProp"
			elif "Conv/BatchNorm/beta" in key:
				vname=overhead+"tpu_batch_normalization/beta"

			elif "Conv/weights/RMSProp_1" in key:
				vname=overhead+"conv2d/kernel/RMSProp_1"
			elif "Conv/weights/RMSProp" in key:
				vname=overhead+"conv2d/kernel/RMSProp"
			elif "Conv/weights/ExponentialMovingAverage" in key:
				vname=overhead+"conv2d/kernel/ExponentialMovingAverage"
			elif "Conv/weights" in key:
				vname=overhead+"conv2d/kernel"
			new_val=reader2.get_tensor(vname)
			var = tf.Variable(new_val, name=key)
			n0=n0+1
		elif "/Conv_1/" in key:
			overhead="efficientnet-b0/head/"
			#Conv_1
			if "/Conv_1/BatchNorm/moving_variance/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/moving_variance/ExponentialMovingAverage"
			elif "Conv_1/BatchNorm/moving_variance" in key:
				vname=overhead+"tpu_batch_normalization/moving_variance"
			elif "/Conv_1/BatchNorm/moving_mean/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/moving_mean/ExponentialMovingAverage"
			elif "Conv_1/BatchNorm/moving_mean" in key:
				vname=overhead+"tpu_batch_normalization/moving_mean"

			elif "Conv_1/BatchNorm/gamma/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/gamma/ExponentialMovingAverage"
			elif "Conv_1/BatchNorm/gamma/RMSProp_1" in key:
				vname=overhead+"tpu_batch_normalization/gamma/RMSProp_1"
			elif "Conv_1/BatchNorm/gamma/RMSProp" in key:
				vname=overhead+"tpu_batch_normalization/gamma/RMSProp"
			elif "Conv_1/BatchNorm/gamma" in key:
				vname=overhead+"tpu_batch_normalization/gamma"

			elif "Conv_1/BatchNorm/beta/ExponentialMovingAverage" in key:
				vname=overhead+"tpu_batch_normalization/beta/ExponentialMovingAverage"
			elif "Conv_1/BatchNorm/beta/RMSProp_1" in key:
				vname=overhead+"tpu_batch_normalization/beta/RMSProp_1"
			elif "Conv_1/BatchNorm/beta/RMSProp" in key:
				vname=overhead+"tpu_batch_normalization/beta/RMSProp"
			elif "Conv_1/BatchNorm/beta" in key:
				vname=overhead+"tpu_batch_normalization/beta"

			elif "Conv_1/weights/RMSProp_1" in key:
				vname=overhead+"conv2d/kernel/RMSProp_1"
			elif "Conv_1/weights/RMSProp" in key:
				vname=overhead+"conv2d/kernel/RMSProp"
			elif "Conv_1/weights/ExponentialMovingAverage" in key:
				vname=overhead+"conv2d/kernel/ExponentialMovingAverage"
			elif "Conv_1/weights" in key:
				vname=overhead+"conv2d/kernel"
			new_val=reader2.get_tensor(vname)
			var = tf.Variable(new_val, name=key)
			n0=n0+1
		elif "global_step" in key:
			vname="global_step"
			global_step=reader2.get_tensor(vname)
			new_val=reader2.get_tensor(vname)
			var = tf.Variable(new_val, name=key)
			n0=n0+1
	#logits
	for key in var_to_shape_map:
		if "Logits" in key:
			if "Conv2d_1c_1x1/weights/RMSProp_1" in key:
				vname="efficientnet-b0/head/dense/kernel/RMSProp_1"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1,tmp.shape[0]))
				tmp=np.insert(tmp,0,append_zeros,axis=1)
				tmp=np.expand_dims(tmp,axis=0)
				tmp=np.expand_dims(tmp,axis=0)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/weights/RMSProp" in key:
				vname="efficientnet-b0/head/dense/kernel/RMSProp"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1,tmp.shape[0]))
				tmp=np.insert(tmp,0,append_zeros,axis=1)
				tmp=np.expand_dims(tmp,axis=0)
				tmp=np.expand_dims(tmp,axis=0)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/weights/ExponentialMovingAverage" in key:
				vname="efficientnet-b0/head/dense/kernel/ExponentialMovingAverage"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1,tmp.shape[0]))
				tmp=np.insert(tmp,0,append_zeros,axis=1)
				tmp=np.expand_dims(tmp,axis=0)
				tmp=np.expand_dims(tmp,axis=0)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/weights" in key:
				vname="efficientnet-b0/head/dense/kernel"
				tmp=reader2.get_tensor(vname)
				print(tmp)
				append_zeros=np.zeros((1,tmp.shape[0]))
				tmp=np.insert(tmp,0,append_zeros,axis=1)
				tmp=np.expand_dims(tmp,axis=0)
				tmp=np.expand_dims(tmp,axis=0)
				print(tmp)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/biases/RMSProp_1" in key:
				vname="efficientnet-b0/head/dense/bias/RMSProp_1"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1))
				tmp=np.insert(tmp,0,append_zeros,axis=0)
				print(tmp.shape)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/biases/RMSProp" in key:
				vname="efficientnet-b0/head/dense/bias/RMSProp"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1))
				tmp=np.insert(tmp,0,append_zeros,axis=0)
				print(tmp.shape)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/biases" in key:
				vname="efficientnet-b0/head/dense/bias"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1))
				tmp=np.insert(tmp,0,append_zeros,axis=0)
				print(tmp.shape)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
			elif "Conv2d_1c_1x1/biases" in key:
				vname="efficientnet-b0/head/dense/bias"
				tmp=reader2.get_tensor(vname)
				append_zeros=np.zeros((1))
				tmp=np.insert(tmp,0,append_zeros,axis=0)
				print(tmp.shape)
				var = tf.Variable(tmp, name=key)
				n0=n0+1
	print(n0)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.save(sess, "./test_ckpt/model.ckpt-"+str(global_step))
