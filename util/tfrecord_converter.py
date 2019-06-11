import tensorflow as tf
import os
import numpy as np
import cv2 

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
 
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
 

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def jpg_to_record(data_path, save_path, label):
    writer = tf.python_io.TFRecordWriter(save_path)
    img_list = os.listdir(data_path)

    for img_path in img_list:
        img = cv2.imread(os.path.join(data_path,img_path))
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = img.tobytes()

        colorspace = 'RGB'
        text = 'None'
        channels = 3
        image_format = 'JPEG'

        example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(224),
      'image/width': _int64_feature(224),
      'image/colorspace': _bytes_feature(bytes(colorspace,encoding='utf8')),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(bytes(os.path.basename(data_path),encoding='utf8')),
      'image/class/text': _bytes_feature(bytes(text,encoding='utf8')),
      'image/object/bbox/xmin': _float_feature(0),
      'image/object/bbox/xmax': _float_feature(0),
      'image/object/bbox/ymin': _float_feature(0),
      'image/object/bbox/ymax': _float_feature(0),
      'image/object/bbox/label': _int64_feature(label),
      'image/format': _bytes_feature(bytes(image_format,encoding='utf8')),
      'image/filename': _bytes_feature(bytes(img_path,encoding='utf8')),
      'image/encoded': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    writer.close()
    print('Finished ',data_path)


def convert(path='/home/yf22/mbv3_pytorch/val_img'):
    dirs = os.listdir(path)
    dirs.sort()
    dirs = np.array(dirs)

    target = [782,282,885,638,689,620,558,623,818,836,167,876,899,493,813,906,744,978,501,600,676,618,461,633,600,438,482,567,664,657,464,488,961,541,415,240,740,784,837,414,827,516,841,680,731,838,585]
    dirs = dirs[target]
    for i in range(len(dirs)):
        data_path = os.path.join(path,dirs[i])
        label = target[i]
        jpg_to_record(data_path,'./augment/train-'+str(i),label)


if __name__ == '__main__':
    convert(path='/home/yf22/mbv3_pytorch/val_img')

		


	
