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
import vgg_16

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("device", "/cpu:*", "device")
#tf.flags.DEFINE_string("device", "/gpu:*", "device")
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

  print(img_list[:10])
  print(annot_list[:10])
  print(FLAGS.batch_size)

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

def init_YOLOBE():
  def init_with_normal():
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
  
  WEs = {
      # step 5 
      "17":tf.get_variable('e_conv_17', shape = [1, 1, 512, 512], initializer=init_with_normal()),
      "18":tf.get_variable('e_conv_18', shape = [3, 3, 512, 1024], initializer=init_with_normal()),
      
      "19":tf.get_variable('e_conv_19', shape = [1, 1, 1024, 512], initializer=init_with_normal()),
      "20":tf.get_variable('e_conv_20', shape = [3, 3, 512, 1024], initializer=init_with_normal()),
      
      "21":tf.get_variable('e_conv_21', shape = [3, 3, 1024, 1024], initializer=init_with_normal()),
      "22":tf.get_variable('e_conv_22', shape = [3, 3, 1024, 1024], initializer=init_with_normal()),
      
      # step 6 
      "23":tf.get_variable('e_conv_23', shape = [3, 3, 1024, 1024], initializer=init_with_normal()),
      "24":tf.get_variable('e_conv_24', shape = [3, 3, 1024, 1024], initializer=init_with_normal()),
      }

  BEs = {
      # step 5 
      "17":tf.get_variable('e_bias_17', shape = [512], initializer=init_with_normal()),
      "18":tf.get_variable('e_bias_18', shape = [1024], initializer=init_with_normal()),
      
      "19":tf.get_variable('e_bias_19', shape = [512], initializer=init_with_normal()),
      "20":tf.get_variable('e_bias_20', shape = [1024], initializer=init_with_normal()),
      
      "21":tf.get_variable('e_bias_21', shape = [1024], initializer=init_with_normal()),
      "22":tf.get_variable('e_bias_22', shape = [1024], initializer=init_with_normal()),
      
      # step 6 
      "23":tf.get_variable('e_bias_23', shape = [1024], initializer=init_with_normal()),
      "24":tf.get_variable('e_bias_24', shape = [1024], initializer=init_with_normal()),
      }

  WFCs = {
      "1":tf.get_variable('fc_1', shape = [FLAGS.num_grid*FLAGS.num_grid*1024, 4096], initializer=init_with_normal()),
      "2":tf.get_variable('fc_2', shape = [4096, FLAGS.num_grid*FLAGS.num_grid*(FLAGS.nclass + 5*FLAGS.B)], initializer=init_with_normal()),
      }

  BFCs = {
      "1":tf.get_variable('fcb_1', shape = [4096], initializer=init_with_normal()),
      "2":tf.get_variable('fcb_2', shape = [FLAGS.num_grid*FLAGS.num_grid*(FLAGS.nclass + 5*FLAGS.B)], initializer=init_with_normal()),
      }

  return WEs, BEs, WFCs, BFCs,

def batch_norm_layer(tensors ,scope_bn, reuse):
  out = tf.contrib.layers.batch_norm(tensors, decay=0.9, center=True, scale=True,
      epsilon=FLAGS.eps,
      updates_collections=None,
      is_training=True,
      reuse=reuse,
      trainable=True,
      data_format='NCHW',
      scope=scope_bn)
  return out

def leaky_relu(tensor):
  return tf.maximum(tensor*0.1, tensor)

def conv_relu(tensor, W, B, name, reuse):
  conved = tf.nn.conv2d(tensor, W[name], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  biased = tf.nn.bias_add(conved, B[name], data_format='NCHW')

#  mean, var = tf.nn.moments(biased, [0, 2, 3], keep_dims=True)
#  normalized = tf.nn.batch_normalization(biased, mean, var, 0, 1, 0.0001)
  normalized = batch_norm_layer(biased, "bne" + name, reuse) 
  relued = leaky_relu(normalized)

  return relued

def model_YOLO(x, WEs, BEs, WFCs, BFCs, drop_prob = 0.5, reuse=False):

  mp_ksize= [1, 1, 2, 2]
  mp_strides=[1, 1, 2, 2]

  # step 5 
  relued = conv_relu(x, WEs, BEs, '17', reuse)
  
  relued = conv_relu(relued, WEs, BEs, '18', reuse)
  
  relued = conv_relu(relued, WEs, BEs, '19', reuse)
  
  relued = conv_relu(relued, WEs, BEs, '20', reuse)
  
  relued = conv_relu(relued, WEs, BEs, '21', reuse)

  relued = conv_relu(relued, WEs, BEs, '22', reuse)

  pooled = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')
 
  # step 6
  relued = conv_relu(pooled, WEs, BEs, '23', reuse)
  
  relued = conv_relu(relued, WEs, BEs, '24', reuse)

  # fully-connected step
  fc = tf.reshape(relued, shape=[FLAGS.batch_size, -1])

  fc = tf.nn.bias_add(tf.matmul(fc, WFCs['1']), BFCs['1'])

  relued = leaky_relu(fc)
  dropouted = tf.nn.dropout(relued, drop_prob)
  
  fc = tf.nn.bias_add(tf.matmul(dropouted, WFCs['2']), BFCs['2'])

  final = tf.reshape(fc, shape=[-1, (FLAGS.nclass + 5*FLAGS.B), 7, 7])

  # no activation or linear activation

  return final

def get_opt(loss, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print "============================"
  print scope
  for item in var_list:
    print item.name
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
#  batch = tf.Variable(0, dtype=tf.int32)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
#  learning_rate = tf.train.exponential_decay(
#      FLAGS.learning_rate,                # Base learning rate.
#      batch,  # Current index into the dataset.
#      1,          # Decay step.
#      FLAGS.weight_decay,                # Decay rate.
#      staircase=True)
  # Use simple momentum for the optimization.
#  optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                         FLAGS.momentum).minimize(loss,
#                                                       var_list=var_list,
#                                                       global_step=batch)
#
#  return optimizer
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
  grads = optimizer.compute_gradients(loss, var_list=var_list)
  return optimizer.apply_gradients(grads)

def calculate_loss(y, out):
  # devide y into each boundbox infos and class info
  yC, yBBs = tf.split(value=y, num_or_size_splits=[FLAGS.nclass, 5*FLAGS.B], axis=3)
  yBBs = tf.split(value=yBBs, num_or_size_splits=[5]*FLAGS.B, axis=3)

  # devide output into each boundbox infos and class info
  C, BBs = tf.split(value=out, num_or_size_splits=[FLAGS.nclass, 5*FLAGS.B], axis=3)
  BBs = tf.split(value=BBs, num_or_size_splits=[5]*FLAGS.B, axis=3)

  # calculate boundbox infos
  lambda_coord = 5
  lambda_noobj = 0.5
 
  coord_term  = 0
  for i in range(FLAGS.B):
    yC, yXY, yWH = tf.split(value=yBBs[i], num_or_size_splits=[1, 2, 2], axis=3)
    C, XY, WH  = tf.split(value=BBs[i], num_or_size_splits=[1, 2, 2], axis=3)

    coord_term += lambda_coord*tf.reduce_sum(yC*tf.square(yXY - XY))
    coord_term += lambda_coord*tf.reduce_sum(yC*tf.square(tf.sqrt(yWH) -tf.sqrt(WH)))
    coord_term += tf.reduce_sum(yC*(yC- C))
    coord_term += lambda_noobj*tf.reduce_sum((1 - yC)*(yC- C))

  # calculate boundbox infos
  obj, _ = tf.split(value=yBBs, num_or_size_splits=[1, 4], axis=3)
  noobj = 1 - obj
  class_term = tf.reduce_sum(obj * tf.square(yC - C))
  
  loss = coord_term + class_term
  return loss

def main(args):
    
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  pretrained = common.load_pretrained("./VGG_16.npy")

  with open(FLAGS.filelist, "r") as f:
    filelist = json.load(f)

  img_list = [os.path.join(FLAGS.train_img_dir, filename + ".png") for filename in filelist]
  annot_list = [os.path.join(FLAGS.train_annot_dir, filename + ".label") for filename in filelist]
  data, label = get_inputs(img_list, annot_list)

  mean = tf.constant(np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32))
  x = tf.cast(data, dtype=tf.float32) - mean

  with tf.device(FLAGS.device):

    x = tf.transpose(x, perm=[0, 3, 1, 2])
    print("0. input setup is done.")


    with tf.variable_scope("vgg_16") as scope:
      Ws, Bs = vgg_16.init_VGG16(pretrained)
    
    with tf.variable_scope("YOLO") as scope:
      WEs, BEs, WFCs, BFCs, = init_YOLOBE()
    
    print("1. variable setup is done.")

    out = vgg_16.model_VGG16(x, Ws, Bs)
    out = model_YOLO(out, WEs, BEs, WFCs, BFCs)
    print("2. model setup is done.")

    out = tf.transpose(out, perm=[0, 2, 3, 1])
    loss = calculate_loss(label, out)
    print("3. loss setup is done.")

    opt = get_opt(loss, "YOLO")
    print("4. optimizer setup is done.")

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

  print("all graph setup is done")

  start = datetime.now()
  print("Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S"))

  config=tf.ConfigProto()
  config.log_device_placement=True
  config.intra_op_parallelism_threads=FLAGS.num_threads
  with tf.Session(config=config) as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
    print("checkpoint: %s" % checkpoint)
    if checkpoint:
      print("Restoring from checkpoint", checkpoint)
      saver.restore(sess, checkpoint)
    else:
      print("Couldn't find checkpoint to restore from. Starting over.")
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(FLAGS.save_dir, filename)

    try:
      for itr in range(FLAGS.max_itrs):

        print('------------------------------------------------------')

        data_val, label_val = sess.run([data, label])

        current = datetime.now()
        print('\telapsed:' + str(current - start))

        if itr > 1 and itr % 10 == 0:
          cv2.imshow('data', cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)
        if itr > 1 and itr % 300 == 0:
          #energy_d_val, loss_d_val, loss_g_val = sess.run([energy_d, loss_d, loss_g])
          print("#######################################################")
          #print "\tE=", energy_d_val, "Ld(x, z)=", loss_d, "Lg(z)=", loss_g
          saver.save(sess, checkpoint)
    except tf.errors.OutOfRangeError:
      print("the last epoch ends.")

    coord.request_stop()
    coord.join(threads)

    cv2.destroyAllWindows()

if __name__ == "__main__":
  tf.app.run()
