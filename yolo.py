import tensorflow as tf
import numpy as np
import glob
import os
import json
from datetime import datetime, date, time
import cv2
import sys
import getopt
import common

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")
#tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("B", "2", "number of Bound Box in grid cell")
tf.flags.DEFINE_integer("num_grid", "7", "number of grids vertically, horizontally")
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_integer("channel", "3", "batch size for training")
tf.flags.DEFINE_integer("img_size", "448", "sample image size")
tf.flags.DEFINE_integer("g_ch_size", "64", "channel size in last discriminator layer")
tf.flags.DEFINE_integer("d_ch_size", "32", "channel size in last discriminator layer")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("save_dir", "yolo_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")


def get_inputs(img_list, annot_list):


  # reference: http://stackoverflow.com/questions/34783030/saving-image-files-in-tensorflow

  print img_list[:10]
  print annot_list[:10]
  print FLAGS.batch_size

  img_queue = tf.train.string_input_producer(img_list, shuffle=False)
  reader = tf.WholeFileReader()
  key, value = reader.read(img_queue)
  decoded = tf.image.decode_png(value)

  annot_queue = tf.train.string_input_producer(annot_list, shuffle=False)
  cell_info_dim = FLAGS.nclass + FLAGS.B*(1 + 4)
  label_bytes = 4*FLAGS.num_grid*FLAGS.num_grid*cell_info_dim
  reader = tf.FixedLengthRecordReader(record_bytes=label_bytes)
  key, value = reader.read(annot_queue)
  annot = tf.decode_raw(value, tf.float32)
  annot = tf.reshape(annot, [FLAGS.num_grid, FLAGS.num_grid, -1])

  return tf.train.shuffle_batch([decoded, annot],
                                 batch_size=FLAGS.batch_size,
                                 num_threads=FLAGS.num_threads,
                                 capacity=FLAGS.batch_size*200,
                                 min_after_dequeue=FLAGS.batch_size*100,
                                 shapes=[[FLAGS.img_size, FLAGS.img_size, FLAGS.channel], [FLAGS.num_grid, FLAGS.num_grid, cell_info_dim]]
                                 )
def init_VGG16(pretrained):
  def load_weight(name):
    print pretrained[name]['weights'].shape
    return tf.constant_initializer(value=pretrained[name]['weights'])

  def load_bias(name):
    print pretrained[name]['biases'].shape
    return tf.constant_initializer(value=pretrained[name]['biases'])

  # initialize with Pretrained model
  Ws = {
      "1_1":tf.get_variable('conv1_1', shape = [3, 3, FLAGS.channel, 64], initializer=load_weight('conv1_1')),
      "1_2":tf.get_variable('conv1_2', shape = [3, 3, 64, 64], initializer=load_weight('conv1_2')),

      "2_1":tf.get_variable('conv2_1', shape = [3, 3, 64, 128], initializer=load_weight('conv2_1')),
      "2_2":tf.get_variable('conv2_2', shape = [3, 3, 128, 128], initializer=load_weight('conv2_2')),

      "3_1":tf.get_variable('conv3_1', shape = [3, 3, 128, 256], initializer=load_weight('conv3_1')),
      "3_2":tf.get_variable('conv3_2', shape = [3, 3, 256, 256], initializer=load_weight('conv3_2')),
      "3_3":tf.get_variable('conv3_3', shape = [3, 3, 256, 256], initializer=load_weight('conv3_3')),

      "4_1":tf.get_variable('conv4_1', shape = [3, 3, 256, 512], initializer=load_weight('conv4_1')),
      "4_2":tf.get_variable('conv4_2', shape = [3, 3, 512, 512], initializer=load_weight('conv4_2')),
      "4_3":tf.get_variable('conv4_3', shape = [3, 3, 512, 512], initializer=load_weight('conv4_3')),

      "5_1":tf.get_variable('conv5_1', shape = [3, 3, 512, 512], initializer=load_weight('conv5_1')),
      "5_2":tf.get_variable('conv5_2', shape = [3, 3, 512, 512], initializer=load_weight('conv5_2')),
      "5_3":tf.get_variable('conv5_3', shape = [3, 3, 512, 512], initializer=load_weight('conv5_3')),
      }

  Bs = {
      "1_1":tf.get_variable('bias1_1', shape = [64], initializer=load_bias('conv1_1')),
      "1_2":tf.get_variable('bias1_2', shape = [64], initializer=load_bias('conv1_2')),

      "2_1":tf.get_variable('bias2_1', shape = [128], initializer=load_bias('conv2_1')),
      "2_2":tf.get_variable('bias2_2', shape = [128], initializer=load_bias('conv2_2')),

      "3_1":tf.get_variable('bias3_1', shape = [256], initializer=load_bias('conv3_1')),
      "3_2":tf.get_variable('bias3_2', shape = [256], initializer=load_bias('conv3_2')),
      "3_3":tf.get_variable('bias3_3', shape = [256], initializer=load_bias('conv3_3')),

      "4_1":tf.get_variable('bias4_1', shape = [512], initializer=load_bias('conv4_1')),
      "4_2":tf.get_variable('bias4_2', shape = [512], initializer=load_bias('conv4_2')),
      "4_3":tf.get_variable('bias4_3', shape = [512], initializer=load_bias('conv4_3')),

      "5_1":tf.get_variable('bias5_1', shape = [512], initializer=load_bias('conv5_1')),
      "5_2":tf.get_variable('bias5_2', shape = [512], initializer=load_bias('conv5_2')),
      "5_3":tf.get_variable('bias5_3', shape = [512], initializer=load_bias('conv5_3')),
      }
  return Ws, Bs

def init_weights():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.1)

  WEs = {

      "6":tf.get_variable('e_conv_6', shape = [7, 7, 512, 4096], initializer=init_with_normal()),
      "7":tf.get_variable('e_conv_7', shape = [1, 1, 4096, 4096], initializer=init_with_normal()),

      "8":tf.get_variable('e_conv_8', shape = [1, 1, 4096, FLAGS.nclass], initializer=init_with_normal()),
      "9":tf.get_variable('e_conv_9', shape = [1, 1, 512, FLAGS.nclass], initializer=init_with_normal()),
      "10":tf.get_variable('e_conv_10', shape = [1, 1, 256, FLAGS.nclass], initializer=init_with_normal()),
      }

  BEs = {

      "6":tf.get_variable('e_bias_6', shape = [4096], initializer=init_with_normal()),
      "7":tf.get_variable('e_bias_7', shape = [4096], initializer=init_with_normal()),
      }

  WDs = {
      "1":tf.get_variable('d_conv_1', shape = [4, 4, FLAGS.nclass, FLAGS.nclass], initializer=init_with_normal()),
      "2":tf.get_variable('d_conv_2', shape = [4, 4, FLAGS.nclass, FLAGS.nclass], initializer=init_with_normal()),
      "3":tf.get_variable('d_conv_3', shape = [16, 16, FLAGS.nclass, FLAGS.nclass], initializer=init_with_normal()),
      }
  return WEs, BEs, WDs

def model_FCN8S(x, y, Ws, Bs, WEs, BEs, WDs, drop_prob = 0.5):

  mp_ksize= [1, 1, 2, 2]
  mp_strides=[1, 1, 2, 2]

  relued = conv_relu(x, Ws['1_1'], Bs['1_1'])
  relued = conv_relu(relued, Ws['1_2'], Bs['1_2'])
  pooled1 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled1, Ws['2_1'], Bs['2_1'])
  relued = conv_relu(relued, Ws['2_2'], Bs['2_2'])
  pooled2 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled2, Ws['3_1'], Bs['3_1'])
  relued = conv_relu(relued, Ws['3_2'], Bs['3_2'])
  relued = conv_relu(relued, Ws['3_3'], Bs['3_3'])
  pooled3 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled3, Ws['4_1'], Bs['4_1'])
  relued = conv_relu(relued, Ws['4_2'], Bs['4_2'])
  relued = conv_relu(relued, Ws['4_3'], Bs['4_3'])
  pooled4 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled4, Ws['5_1'], Bs['5_1'])
  relued = conv_relu(relued, Ws['5_2'], Bs['5_2'])
  relued = conv_relu(relued, Ws['5_3'], Bs['5_3'])
  pooled5 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled5, WEs['6'], BEs['6'])
  dropouted = tf.nn.dropout(relued, drop_prob)

  relued = conv_relu(dropouted, WEs['7'], BEs['7'])
  dropouted = tf.nn.dropout(relued, drop_prob)

  score_fr = tf.nn.conv2d(dropouted, WEs['8'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

  score_pool4 = tf.nn.conv2d(pooled4, WEs['9'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  shape_list = tf.shape(score_pool4)
  out_shape = tf.stack(shape_list)
  upscore2 = tf.nn.conv2d_transpose(score_fr, WDs['1'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  fuse_pool4 = tf.add(upscore2, score_pool4)

  score_pool3 = tf.nn.conv2d(pooled3, WEs['10'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  shape_list = tf.shape(score_pool3)
  out_shape = tf.stack(shape_list)
  upscore_pool4 = tf.nn.conv2d_transpose(fuse_pool4, WDs['2'], out_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
  fuse_pool3 = tf.add(upscore_pool4, score_pool3)

  shape_list = tf.shape(y)
  out_shape = tf.stack(shape_list)
  upscore_pool8 = tf.nn.conv2d_transpose(fuse_pool3, WDs['3'], out_shape, strides=[1, 1, 8, 8], padding='SAME', data_format='NCHW')

  final_score = upscore_pool8

  return final_score

def main(args):
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  with open(FLAGS.filelist, "r") as f:
    filelist = json.load(f)

  img_list = [os.path.join(FLAGS.train_img_dir, filename + ".png") for filename in filelist]
  annot_list = [os.path.join(FLAGS.train_annot_dir, filename + ".label") for filename in filelist]

  print img_list[:10]
  print annot_list[:10]

  pretrained = common.load_pretrained("./VGG_16.npy")

#  with tf.variable_scope("VGG16") as scope:
#    Ws, Bs = init_VGG16(pretrained)
#  with tf.variable_scope("FCN8S") as scope:
#    WEs, BEs, WDs = init_weights()

  W = tf.get_variable('test', shape = [4, 4, FLAGS.nclass, FLAGS.nclass]),
  data, label = get_inputs(img_list, annot_list)

  #casted = tf.cast(decoded, tf.float32)
  x = tf.transpose(data, perm=[0, 3, 1, 2])
  y = tf.transpose(label, perm=[0, 3, 1, 2])

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  start = datetime.now()
  print "Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S")

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=FLAGS.num_threads)) as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
    print "checkpoint: %s" % checkpoint
    if checkpoint:
      print "Restoring from checkpoint", checkpoint
      saver.restore(sess, checkpoint)
    else:
      print "Couldn't find checkpoint to restore from. Starting over."
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(FLAGS.save_dir, filename)

    try:
      for itr in range(FLAGS.max_itrs):

        print "------------------------------------------------------"

        data_val, label_val = sess.run([data, label])

        current = datetime.now()
        print "\telapsed:", current - start

        if itr > 1 and itr % 10 == 0:
          cv2.imshow('data', cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)
        if itr > 1 and itr % 300 == 0:
          #energy_d_val, loss_d_val, loss_g_val = sess.run([energy_d, loss_d, loss_g])
          print "#######################################################"
          #print "\tE=", energy_d_val, "Ld(x, z)=", loss_d, "Lg(z)=", loss_g
          saver.save(sess, checkpoint)
    except tf.errors.OutOfRangeError:
      print "the last epoch ends."

    coord.request_stop()
    coord.join(threads)

    cv2.destroyAllWindows()

if __name__ == "__main__":
  tf.app.run()
