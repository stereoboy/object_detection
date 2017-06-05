import tensorflow as tf
import numpy as np
#import multiprocessing as mp
import glob
import os
import json
from datetime import datetime, date, time
import cv2
import sys
import getopt
import common
import vgg_16
import random
from PIL import Image
import image_process as improc

FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string("device", "/cpu:*", "device")
tf.flags.DEFINE_string("device", "/gpu:*", "device")
tf.flags.DEFINE_integer("max_epoch", "200", "maximum iterations for training")
#tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_integer("num_anchor", "5", "number of anchor per grid cell")
tf.flags.DEFINE_integer("num_out_layer", "2", "number of output layers")
tf.flags.DEFINE_integer("num_grid", "13", "number of grids vertically, horizontally")
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_float("confidence", "0.1", "confidence limit")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_integer("channel", "3", "batch size for training")
tf.flags.DEFINE_integer("img_orig_size", "646", "sample image size")
tf.flags.DEFINE_integer("img_size", "416", "sample image size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("save_dir", "yolov2_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")

slim = tf.contrib.slim

def leaky_relu(tensor):
  return tf.maximum(tensor*0.1, tensor)

def base_conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format='NCHW', scope=None):

  out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, data_format=data_format, activation_fn=None, normalizer_fn=None, scope=scope)
  out = slim.batch_norm(out, activation_fn=leaky_relu, scope='bn_' + scope)

  return out

def model_backend(_out0, _out1):

  # re-orgarnize from large image to small image
  # 2x2x1 -> 1x1x4
  def reorganize(x):
    a = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    a = np.expand_dims(a, axis=2)
    print('a', a.shape)
    print('a', a)
    kernel = tf.Variable(a, trainable=False)
    out = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')
    return out

  out0 = base_conv2d(_out0, 1024, [3, 3], scope='backend1')
  out0 = reorganize(out0)

  out1 = base_conv2d(_out1, 1024, [3, 3], scope='backend1')
  out1 = base_conv2d(out1, 1024, [3, 3], scope='backend2')

  out = tf.concat([out0, out1], axis=3)

  out = model_detect(out)

  detect_out = FLAGS.num_anchor*(FLAGS.nclass + 1 + 4)

  out = base_conv2d(_out, 1024, [3, 3], scope='detect0')
  out = base_conv2d(_out, detect_out, [1, 1], scope='detect1')

  return out

def calculate_loss(y, out):

  loss = 

  return loss

def main(args):

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  vgg_16.setup_vgg_16()

  _R_MEAN = 123.68
  _G_MEAN = 116.78
  _B_MEAN = 103.94


  with tf.Graph().as_default(): 
    mean = tf.constant(np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32))

    drop_prob = tf.placeholder(tf.float32)
    _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel])
    _st = tf.placeholder(tf.float32, [None, 4])
    _flip = tf.placeholder(tf.bool, [None])

    aug = improc.augment_scale_translate_flip(_x, FLAGS.img_size, _st, _flip, FLAGS.batch_size)
    aug = tf.map_fn(lambda x:improc.augment_br_sat_hue_cont(x), aug)
    x = tf.cast(aug, dtype=tf.float32) - mean
    x = improc.augment_gaussian_noise(x)

    x = tf.transpose(x, perm=[0, 3, 1, 2])

    test_input = tf.range(start=0, limit=(4*4), dtype=tf.float32)
    test_input = tf.reshape(test_input, (1, 4, 4, 1))
    a = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    a = np.expand_dims(a, axis=2)
    print('a', a.shape)
    print('a', a)
    kernel = tf.Variable(a, trainable=False)
    conved = tf.nn.conv2d(test_input, kernel, strides=[1, 2, 2, 1], padding='SAME')
    with slim.arg_scope(vgg_16.vgg_arg_scope()):
      outputs, end_points = vgg_16.vgg_16_base(x)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(vgg_16.checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    print("==get_opt()============================")
    for item in var_list:
      print(item.name)

#    tensor = tf.contrib.framework.get_variables_by_name('vgg_16/conv1/conv1_1/weights')
#    print('....:', tensor[0].get_shape())

    config=tf.ConfigProto()
    #config.log_device_placement=True
    config.intra_op_parallelism_threads=FLAGS.num_threads
    with tf.Session(config=config) as sess:

      saver = tf.train.Saver()
      checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
      print("checkpoint: %s" % checkpoint)
      if checkpoint:
        print("Restoring from checkpoint: %s" % checkpoint)
        saver.restore(sess, checkpoint)
      else:
        print("Couldn't find checkpoint to restore from. Starting over.")
        dt = datetime.now()
        filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = os.path.join(FLAGS.save_dir, filename)

      init_fn(sess)

      sess.run(init_op)

      #print(tensor[0].eval().shape)
      print(test_input.eval())
      print(kernel.eval())
      print(kernel.eval().shape)
      print(conved.eval())
      print(conved.eval().shape)

#      saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()
