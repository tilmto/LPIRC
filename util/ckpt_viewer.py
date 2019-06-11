import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n=0
with tf.Session() as sess:
    checkpoint_path="model.ckpt-876168"
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        #print(key)
        n=n+1
        #continue
        #if "MobilenetV2/expanded_conv_5" in key:
        #if "expanded_conv_1" in key:
        #print(reader.get_tensor(key))
        if reader.get_tensor(key).shape==(3,3,3,32):
        #if "efficientnet-b0/head/dense/kernel/RMSProp_1" in key:
        #if key=="mnasnet-a1/mnas_net_model/mnas_blocks_5/conv2d_16/kernel":
            #tmp=np.insert(tmp,0,append_zeros,axis=1)
            print("tensor_name: ", key,reader.get_tensor(key).shape)

            #tmp=reader.get_tensor(key)
            #append_zeros=np.zeros((1,tmp.shape[0]))
            #tmp=np.insert(tmp,0,append_zeros,axis=1)

            #tmp=np.expand_dims(tmp,axis=0)
            #tmp=np.expand_dims(tmp,axis=0)
            #print(tmp.shape[0])

            #print(tmp.shape)
            #print(reader.get_tensor(key))
    print (n)
