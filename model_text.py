#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:48:32 2018

@author: hs
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image  
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#label_lines = [line.rstrip() for line in tf.gfile.GFile("model/output_labels.txt")]
def read_and_decode(tfrecord_file_path):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([tfrecord_file_path],shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data':tf.FixedLenFeature([256*256],tf.float32),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'id' : tf.FixedLenFeature([], tf.int64),
                                       })
    image=features['data']
    image=tf.reshape(image,[256,256])
    label = tf.cast(features['label'], tf.int64)
    num= tf.cast(features['id'], tf.int64)
    return image,label,num

def model_text(tfrecord_path):
    new_folder_path='text_image_hs'
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    image,label,num=read_and_decode(tfrecord_path)   
    im_num=0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
       im_num += 1 
    print('num of text image:'+str(im_num))
    
    # Unpersists graph from file
    with tf.gfile.FastGFile("1-8000.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        #启动多线程
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
        #tf.train.start_queue_runners(sess=sess)
        w=0
        L=[]
        L_true=[]
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for i in range(im_num):
            a,b,c=sess.run([image,label,num])
            img=(a+1)*128
            im=Image.fromarray(img).convert('L')
            im.save(new_folder_path+'/'+str(c)+'_''Label_'+str(b)+'.jpg')#存下图片
            image_path = new_folder_path+'/'+str(c)+'_''Label_'+str(b)+'.jpg'
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})   
            label_p=tf.arg_max(predictions,1)
            label_p=sess.run(label_p)
            l=int(label_p)+1
            L.append(l)
            L_true.append(int(b))
            if int(l)!=int(b):
                print('image-'+str(c)+' is wrong!  '+'wrong label:'+str(l)+'   true label:'+str(b))
                w+=1
        coord.request_stop()
        coord.join(threads)
    return L,L_true

def main():
    
    label=model_text('TFcordX_text.tfrecord')
    '''
    tfrecord_path='data/TFcodeX_10.tfrecord'
    label,l=model_text(tfrecord_path)
    correct_prediction=tf.equal(label,l)
    accuary=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    with tf.Session() as sess:
        a=sess.run(accuary)
        print('accuary is:'+str(a))
    '''
    return label
    

if __name__ == '__main__':
    main()

    
    




  

