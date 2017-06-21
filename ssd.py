import tensorflow as tf
import numpy as np
#import multiprocessing as mp
import glob
import os
import json
from datetime import datetime, date, time
import cv2
import sys
import voc
import utils
import common
import vgg_16
import random
from PIL import Image
import image_process as improc

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string("device", "/cpu:*", "device")
tf.flags.DEFINE_string("device", "/gpu:*", "device")
tf.flags.DEFINE_integer("max_epoch", "200", "maximum iterations for training")
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_integer("nclass", "21", "class num")
tf.flags.DEFINE_float("iou_threshold", "0.5", "threshold for jaccard overlay(iou)")
tf.flags.DEFINE_float("confidence", "0.1", "confidence limit")
tf.flags.DEFINE_float("negative_ratio", "3.0", "ratio between negative and positive samples")
tf.flags.DEFINE_float("learning_rate0", "1e-3", "Learning rate for Optimizer")
tf.flags.DEFINE_float("learning_rate1", "1e-4", "Learning rate for Optimizer")
tf.flags.DEFINE_float("learning_rate2", "1e-5", "Learning rate for Optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_integer("channel", "3", "batch size for training")
tf.flags.DEFINE_integer("img_orig_size", "646", "sample image size")
tf.flags.DEFINE_integer("img_size", "300", "sample image size")
tf.flags.DEFINE_integer("img_vis_size", "428", "sample image size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("balanced_filelist", "balanced.json", "normalized filelist")
tf.flags.DEFINE_string("save_dir", "ssd_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("log_dir", "ssd_logs", "directory for log")
tf.flags.DEFINE_string("log_name", "ssd", "directory for log")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")
tf.flags.DEFINE_float("weight_decay", "0.0005", "weight decay for L2 regularization")
tf.flags.DEFINE_float("init_stddev", "0.1", "stddev for initializer")
tf.flags.DEFINE_bool("dbprint", "False", "option for debug print")

slim = tf.contrib.slim

def visualization(img, annots, anchor_infos, idx2obj, palette, options=['draw_anchor', 'target']):
  #print("visualization()")
  h, w = img.shape[:2]

  vis_grid = np.zeros_like(img)

  # iterate layers
  for i, (layer_dim, anchor_scales) in enumerate(anchor_infos):
    (h_num_grid, w_num_grid) = layer_dim
    box_dim = (FLAGS.nclass + 4)*len(anchor_scales)

    annot = annots[i]
    for row in range(h_num_grid):
      for col in range(w_num_grid):
        offset = 0
        for j, anchor_scale in enumerate(anchor_scales):
          conf = annot[row, col, offset:offset + FLAGS.nclass]
          idx = int(np.argmax(conf))
          if 'target' in options:
            max_conf = np.max(conf)
          else:
            max_conf = np.max(common.softmax(conf))
#            if np.max(conf) > 0.5:
#              print(conf)
#              print(common.softmax(conf))

          if idx > 0 and max_conf > 0.5:
            #print(idx, conf)
            _color = palette[idx]
            #color = (int(_color[2]), int(_color[1]), int(_color[0]))
            color = _color[::-1].astype(dtype=np.int32).tolist()
            name = idx2obj[idx]

            anchor_w, anchor_h = anchor_scale
            anchor_cx, anchor_cy = (col + .5)/w_num_grid, (row  + .5)/h_num_grid
            anchor_cwh = ((anchor_cx, anchor_cy), (anchor_w, anchor_h))
            anchor_bbox = improc.cvt_cwh2bbox(anchor_cwh)

            b = offset + FLAGS.nclass
            e = b + 4
            reg_cx, reg_cy, reg_nw, reg_nh = annot[row, col, b:e]

            cx, cy = reg_cx*anchor_w + anchor_cx, reg_cy*anchor_h + anchor_cy
            nw, nh = np.exp(reg_nw)*anchor_w, np.exp(reg_nh)*anchor_h

            cwh = ((cx, cy), (nw, nh))
#            print(cwh)
#            print(anchor_w, anchor_h)
#            print(reg_cx, reg_cy, reg_nw, reg_nh)
#            print('--------------------------------------')
            ((x1, y1), (x2, y2)) = improc.cvt_cwh2bbox(cwh)
            name = name + '%d_%d'%(i, j) + '_%.2f'%(np.max(conf))

#          print("{} is located at ({}, {})".format(name, row, col))
#          print("WxH at {}x{}".format(bw, bh))

            b = (int(x1*w), int(y1*h))
            e = (int(x2*w), int(y2*h))

            anchor_b = (int(anchor_bbox[0][0]*w), int(anchor_bbox[0][1]*h))
            anchor_e = (int(anchor_bbox[1][0]*w), int(anchor_bbox[1][1]*h))

            #vis_grid= cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
            if 'draw_anchor' in options:
              img = cv2.rectangle(img, anchor_b, anchor_e, (0, 0, 255), 1)
            img = cv2.rectangle(img, b, e, color, 6)
            img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            img = cv2.circle(img, (int(cx*w), int(cy*h)), 4, color, -1)

          offset += (FLAGS.nclass + 4)

  return img

def build_feed_annots(_feed_annots, anchor_infos):
  print('build_feed_annots()')
  batch_size = len(_feed_annots)

  # scale and translation
  # input image is resized 537x537
  # we will choose random crop size and offset, and resize into 428x428
  # This is equivalent to resize ~20% and crop with the widnow of 428x428
  scales = np.random.uniform(0.70, 0.90, [batch_size, 1])
  offsets = (1 - scales)*np.random.uniform(0.0, 1.0, [batch_size, 2])
  ends = offsets + scales
  feed_scaletrans = np.concatenate([offsets, ends], axis=1)
  feed_flips = np.random.randint(0, 2, [batch_size])

  feed_annots_list = []

  # iterate layers
  for layer_dim, anchor_scales in anchor_infos:
    (h_num_grid, w_num_grid) = layer_dim
    box_dim = (FLAGS.nclass + 4)*len(anchor_scales)

    feed_annots = np.zeros((batch_size, h_num_grid, w_num_grid, box_dim), np.float32)

    # fill all crid as blank
    for i in range(len(anchor_scales)):
      feed_annots[:, :, :, (FLAGS.nclass + 4)*i] = 1.0

    # build augmented annotations
    for i, _annot in enumerate(_feed_annots):
      # each image
      annot = _annot.copy()
      _w, _h = _annot[0, :2]
      scale = scales[i, 0] # w, h scale is same
      w, h = _w*scale, _h*scale
      #print ('scale:{}'.format(scale))
      (_offset_y, _offset_x) = offsets[i]
      #print('_offset:{}, {}'.format(_offset_x, _offset_y))
      #print(_w, _h)
      (offset_x, offset_y) = (_w*_offset_x, _h*_offset_y)
      #print('offset:{}, {}'.format(offset_x, offset_y))
      #print(_annot)
      annot[1:, 1:3] = _annot[1:, 1:3] - offset_x
      annot[1:, 3:5] = _annot[1:, 3:5] - offset_y
      #print( annot)

      annot[1:, 1:3] = annot[1:, 1:3]
      annot[1:, 3:5] = annot[1:, 3:5]

      for box in annot[1:]:
        idx, x1, x2, y1, y2 = box
        x1, x2 = np.max((0.0, x1)), np.min((w, x2))
        y1, y2 = np.max((0.0, y1)), np.min((h, y2))

        if feed_flips[i]:
          x1, x2 = w - x1, w - x2
          tmp = x1
          x1 = x2
          x2 = tmp

        bbox = ((x1/w, y1/h), (x2/w, y2/h))
        ((cx, cy), (nw, nh)) = improc.cvt_bbox2cwh(bbox)
        x_loc, y_loc = int(cx*w_num_grid), int(cy*h_num_grid)
        idx = int(idx)

        if x_loc < 0 or y_loc < 0 or x_loc >= w_num_grid or y_loc >= h_num_grid:
          continue
        if nw < 0 or nh < 0:
          continue

        offset = 0
        for anchor_scale in anchor_scales:
          anchor_w, anchor_h = anchor_scale
          anchor_cx, anchor_cy = (x_loc + .5)/w_num_grid, (y_loc + .5)/h_num_grid
          anchor_cwh = ((anchor_cx, anchor_cy), (anchor_w, anchor_h))
          # iou with the current anchor region is > 0.5
          anchor_bbox = improc.cvt_cwh2bbox(anchor_cwh)

          iou = improc.cal_iou(bbox, anchor_bbox)
          if iou > FLAGS.iou_threshold:
            class_array = np.zeros((FLAGS.nclass))
            class_array[idx] = 1.0
            feed_annots[i, y_loc, x_loc, offset:offset + FLAGS.nclass] = class_array

            b = offset + FLAGS.nclass
            e = b + 4
            reg_cx, reg_cy = (cx - anchor_cx)/anchor_w, (cy - anchor_cy)/anchor_h
            reg_nw, reg_nh = np.log(nw/anchor_w), np.log(nh/anchor_h)
            feed_annots[i, y_loc, x_loc, b:e] = np.array((reg_cx, reg_cy, reg_nw, reg_nh), np.float32)
            if i == 0:
              utils.debug_print(FLAGS.dbprint, 'iou:', iou)
              utils.debug_print(FLAGS.dbprint, 'bbox:', bbox)
              utils.debug_print(FLAGS.dbprint, 'anchor_bbox:', anchor_bbox)
              utils.debug_print(FLAGS.dbprint, (anchor_w, anchor_h), iou)
              utils.debug_print(FLAGS.dbprint, '--------------------------------------')


          offset += (FLAGS.nclass + 4)

    feed_annots_list.append(feed_annots)

  return feed_scaletrans, feed_flips, feed_annots_list


# 1.0 means total image width
def init_anchor_scales(num_layers):

  min_scale = 0.2
  max_scale = 0.9

  scale_list = []
  for k in range(1, num_layers + 1):
    scales = []
    s_k = min_scale + (max_scale - min_scale)/(num_layers - 1)*(k - 1)
    s_k_1 =   min_scale + (max_scale - min_scale)/(num_layers - 1)*k
    scales.append((s_k, s_k))
    scales.append((np.sqrt(s_k*s_k_1), np.sqrt(s_k*s_k_1)))
    scales.append((s_k*np.sqrt(2.0), s_k*np.sqrt(0.5)))
    scales.append((s_k*np.sqrt(0.5), s_k*np.sqrt(2.0)))
    # ...
    if 2 <= k <= 4:
      scales.append((s_k*np.sqrt(3.0), s_k*np.sqrt(1.0/3.0)))
      scales.append((s_k*np.sqrt(1.0/3.0), s_k*np.sqrt(3.0)))
    scale_list.append(scales)
  return scale_list

def base_conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format='NCHW', scope=None):

  out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, data_format=data_format, activation_fn=None, normalizer_fn=None, scope=scope)
  out = slim.batch_norm(out, activation_fn=tf.nn.relu, scope=scope+'_bn', is_training=True)

  return out

def model_ssd(frontend):

  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                      #biases_initializer=tf.zeros_initializer(),
                      data_format='NCHW'):
    # 10 x 10
    conv6_1 = slim.conv2d(frontend, 256, [1, 1], stride=1, padding='SAME', scope='conv6_1')
    conv6_2 = slim.conv2d(conv6_1, 512, [3, 3], stride=2, padding='SAME', scope='conv6_2')

    # 5 x 5
    conv7_1 = slim.conv2d(conv6_2, 256, [1, 1], stride=1, padding='SAME', scope='conv7_1')
    conv7_2 = slim.conv2d(conv7_1, 512, [3, 3], stride=2, padding='SAME', scope='conv7_2')

    # 3 x 3
    conv8_1 = slim.conv2d(conv7_2, 256, [1, 1], stride=1, padding='SAME', scope='conv8_1')
    conv8_2 = slim.conv2d(conv8_1, 512, [3, 3], stride=2, padding='SAME', scope='conv8_2')

    # 1 x 1
    conv9_1 = slim.conv2d(conv8_2, 256, [1, 1], stride=1, padding='SAME', scope='conv9_1')
    conv9_2 = slim.conv2d(conv9_1, 512, [3, 3], stride=2, padding='VALID', scope='conv9_2')

  return conv6_2, conv7_2, conv8_2, conv9_2

def model_backend(out_layers, anchor_scales_list):

  ret_layers = []
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                      #biases_initializer=tf.zeros_initializer(),
                      data_format='NCHW'):
    for i, (layer, anchor_scales) in enumerate(zip(out_layers, anchor_scales_list)):
      out_size = (FLAGS.nclass + 4)*len(anchor_scales)

      _out = slim.conv2d(layer, out_size, [3, 3], stride=1, padding='SAME', data_format='NCHW', scope='backend{}'.format(i))
      out = tf.transpose(_out, perm=[0, 2, 3, 1])
      ret_layers.append(out)

  return ret_layers

def smooth_l1_loss(x):

  cond = tf.abs(x) < 1.0
  square = 0.5*tf.square(x)
  absolute = tf.abs(x) - 0.5
  return tf.where(cond, square, absolute)

def calculate_loss(ys, outs, anchor_scales_list):

  flat_class_y   = []
  flat_class_out = []

  flat_coord_y   = []
  flat_coord_out = []

  for i in range(len(ys)):
    y, out, anchor_scales = ys[i], outs[i], anchor_scales_list[i]

    with tf.name_scope('anchors{}'.format(i)):
      offset = 0
      print("i:",i, y, out)
      for j, anchor_scale in enumerate(anchor_scales):
        with tf.name_scope('slice{}'.format(j)):
          class_y = y[:, :, :, offset:offset + FLAGS.nclass]
          class_y = tf.reshape(class_y, shape=[FLAGS.batch_size, -1, FLAGS.nclass])

          class_out = out[:, :, :, offset:offset + FLAGS.nclass]
          class_out = tf.reshape(class_out, shape=[FLAGS.batch_size, -1, FLAGS.nclass])

          coord_y = y[:, :, :, offset + FLAGS.nclass: offset + FLAGS.nclass + 4]
          coord_y = tf.reshape(coord_y, shape=[FLAGS.batch_size, -1, 4])

          coord_out = out[:, :, :, offset + FLAGS.nclass: offset + FLAGS.nclass + 4]
          coord_out = tf.reshape(coord_out, shape=[FLAGS.batch_size, -1, 4])

          #class_out = utils.tf_Print(class_out, [class_out, tf.shape(class_out)[1:]], summarize=200, message="class_out{}_{}:".format(i, j))
          flat_class_y.append(class_y)
          flat_class_out.append(class_out)
          flat_coord_y.append(coord_y)
          flat_coord_out.append(coord_out)

        offset += (FLAGS.nclass + 4)

  flat_class_y = tf.concat(flat_class_y, axis=1, name='flat_class_y')
  flat_class_y = utils.tf_Print(FLAGS.dbprint, flat_class_y, summarize=200, message="flat_class_y:")
  flat_class_out = tf.concat(flat_class_out, axis=1, name='flat_class_out')
  flat_class_out = utils.tf_Print(FLAGS.dbprint, flat_class_out, summarize=200, message="flat_class_out:")
  flat_coord_y = tf.concat(flat_coord_y, axis=1, name='flat_coord_y')
  flat_coord_out = tf.concat(flat_coord_out, axis=1, name='flat_coord_out')

  #positive_mask = tf.reduce_max(flat_class_y, axis=2) > 0.5
  #negative_mask = tf.logical_not(positive_mask)

#  negative_mask = tf.cast(flat_class_y[:, :, 0], dtype=tf.bool)
#  positive_mask = tf.logical_not(negative_mask)
#
#  positive_mask = tf.cast(positive_mask, dtype=tf.float32)
#  positive_num  = tf.reduce_sum(positive_mask, axis=1)
#  negative_mask = tf.cast(negative_mask, dtype=tf.float32)
#  negative_num  = tf.reduce_sum(negative_mask, axis=1)
  with tf.name_scope('calculate_masks'):
    negative_mask = flat_class_y[:, :, 0]
    positive_mask = 1.0 - negative_mask

    positive_num  = tf.reduce_sum(positive_mask, axis=1, name='positive_num')
    negative_num  = tf.reduce_sum(negative_mask, axis=1)

    negative_num = tf.cast(tf.minimum(negative_num, positive_num*FLAGS.negative_ratio), dtype=tf.int32, name='negative_num')

    positive_num = utils.tf_Print(FLAGS.dbprint, positive_num, summarize=49, message="positive_num:")
    negative_num = utils.tf_Print(FLAGS.dbprint, negative_num, summarize=49, message="negative_num:")

    positive_num += 1 #dummy to prevent divied by zero
    negative_num += 1 #dummy to prevent set k=zero in top_k

  with tf.name_scope('conf_loss'):
    conf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_class_out, labels=flat_class_y)

    conf_loss = utils.tf_Print(FLAGS.dbprint, conf_loss, summarize=49, message="conf_loss:")

  negative_loss = []
  with tf.name_scope('negative_conf_loss'):
    for i in range(FLAGS.batch_size):
      # use minus for sort
      with tf.name_scope('top_k{}'.format(i)):
        values, indices = tf.nn.top_k((conf_loss[i]*negative_mask[i]), k=negative_num[i])
        #values = utils.tf_Print(FLAGS.dbprint, values, summarize=49, message="values{}:".format(i))
        # recover sign by minus
        negative_loss.append(tf.reduce_sum(values))
    negative_loss = tf.stack(negative_loss)

    negative_loss = utils.tf_Print(FLAGS.dbprint, negative_loss, summarize=49, message="negative_loss:")
  #negative_loss /= FLAGS.batch_size

  with tf.name_scope('positive_conf_loss'):
    positive_loss = tf.reduce_sum(conf_loss*positive_mask, axis=1)
    positive_loss = utils.tf_Print(FLAGS.dbprint, positive_loss, summarize=49, message="positive_loss:")

  with tf.name_scope('coord_loss'):
    loc_loss = smooth_l1_loss(flat_coord_out - flat_coord_y)
    coord_term = tf.reduce_sum(loc_loss*tf.expand_dims(positive_mask, axis=2), axis=[1, 2])
    coord_term = utils.tf_Print(FLAGS.dbprint, coord_term, summarize=49, message="coord_term:")

  loss = positive_loss + negative_loss + coord_term
  loss = utils.tf_Print(FLAGS.dbprint, loss, summarize=49, message="lossC:")

  loss = tf.div(loss, positive_num)
# code below is not working. it makes gradients nan
#  cond = positive_num > 0.0
#  cond = utils.tf_Print(cond, [cond, tf.shape(cond)], summarize=49, message="cond:")
#  a = tf.div(loss, positive_num)
#  a = utils.tf_Print(a, [a, tf.shape(a)], summarize=49, message="a:")
#  b = tf.zeros_like(loss)
#  b = utils.tf_Print(b, [b, tf.shape(b)], summarize=49, message="b:")
#  loss = tf.where(cond, a, b)
  loss = utils.tf_Print(FLAGS.dbprint, loss, summarize=49, message="lossD:")
  loss = tf.reduce_mean(loss)

  tf.summary.scalar('positive_term', tf.reduce_mean(positive_loss))
  tf.summary.scalar('negative_term', tf.reduce_mean(negative_loss))
  tf.summary.scalar('coord_term', tf.reduce_mean(coord_term))

  return loss

def get_opt(loss, scope):

  print('loss:{}'.format(loss))

  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print("==get_opt()============================")
  print(scope)
  for item in var_list:
    print(item.name)
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
#  batch = tf.Variable(0, dtype=tf.int32)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
#  learning_rate = tf.train.exponential_decay(
#      FLAGS.learning_rate,                # Base learning rate.
#      batch,  # Current index into the dataset.
#      1,          # Decay step.
#      FLAGS.weight_decay,                # Decay rate.
#      staircase=True)
  # Use simple momentum for the optimization.

#  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
#                                                 1, 0.998, staircase=True)
#  learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
#  lr_decay_op1 = tf.assign(learning_rate, 1e-4)
#  lr_decay_op2 = tf.assign(learning_rate, 1e-5)

  boundaries = [40000, 50000]
  values = [FLAGS.learning_rate0, FLAGS.learning_rate1, FLAGS.learning_rate2]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
  tf.summary.scalar('learning_rate', learning_rate)
  optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
#  optimizer = tf.train.AdamOptimizer(learning_rate)
  opt = optimizer.minimize(loss, var_list=var_list, global_step=global_step)

#  learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
#  learning_rate = utils.tf_Print(learning_rate, [learning_rate], message="learning_rate:")
#  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
#
#  learning_rate = tf.Variable(
#      float(1e-3), trainable=False, dtype=tf.float32)
#  lr_decay_op1 = learning_rate.assign(1e-3)
#  lr_decay_op2 = learning_rate.assign(1e-4)
#  learning_rate = utils.tf_Print(learning_rate, [learning_rate], message="learning_rate:")

#  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
#                                                 10, 0.9995, staircase=True)
#  learning_rate = utils.tf_Print(learning_rate, [learning_rate], message="learning_rate:")
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
#                                                                var_list=var_list,
#                                                                global_step=global_step)

  return opt, 0, 0
#  return tf.train.AdamOptimizer(0.0001).minimize(loss)
#  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
#  grads = optimizer.compute_gradients(loss, var_list=var_list)
#  return optimizer.apply_gradients(grads)

def main(args):

  colormap, palette = voc.build_colormap_lookup(21)
  idx2obj = voc.idx2obj

  with open(FLAGS.balanced_filelist, "r") as f:
    filelist = json.load(f)['train']

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

  vgg_16.setup_vgg_16()

  _R_MEAN = 123.68
  _G_MEAN = 116.78
  _B_MEAN = 103.94

  log_filename = os.path.join(FLAGS.log_dir, FLAGS.log_name)
  print("summary_log:", log_filename)
  writer = tf.summary.FileWriter(log_filename)

  with tf.Graph().as_default():
    mean = tf.constant(np.array((_R_MEAN, _G_MEAN, _B_MEAN), dtype=np.float32))

    drop_prob = tf.placeholder(tf.float32)
    _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel], name='raw_input')
    _st = tf.placeholder(tf.float32, [None, 4], name='scale_translation_seed')
    _flip = tf.placeholder(tf.bool, [None], name='flip_seed')

    with tf.name_scope('augmentation'):
      aug = improc.augment_scale_translate_flip(_x, FLAGS.img_size, _st, _flip, FLAGS.batch_size)
      aug = improc.augment_br_sat_hue_cont(aug)
      with tf.name_scope('mean_subtraction'):
        mean = tf.constant(np.array((_R_MEAN, _G_MEAN, _B_MEAN), dtype=np.float32))
        x = tf.cast(aug, dtype=tf.float32) - mean
      x = improc.augment_gaussian_noise(x)

    x = tf.transpose(x, perm=[0, 3, 1, 2], name='augmented_input')

    print("0. input image setup is done.")

    with slim.arg_scope(vgg_16.vgg_arg_scope()):
      _, end_points = vgg_16.vgg_16_base(x)

    out_layers = []
    out_layers.append(end_points['vgg_16/conv4/conv4_3'])

    vgg_outs = end_points['vgg_16/conv5/conv5_3']
    out_layers.append(end_points['vgg_16/conv5/conv5_3'])

    #print('vgg_outs', vgg_outs, vgg_outs.get_shape())
    #print('end_points', end_points)
    init_vgg_16_fn = slim.assign_from_checkpoint_fn(
        os.path.join(vgg_16.checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    with tf.variable_scope('ssd') as scope:
      layers = model_ssd(vgg_outs)

    out_layers.extend(layers)
    print('layers', layers)

    anchor_scales_list = init_anchor_scales(len(out_layers))
    with tf.variable_scope('ssd_backend') as scope:
      out_layers = model_backend(out_layers, anchor_scales_list)
      layer_dims = [(int(layer.get_shape()[1]), int(layer.get_shape()[2])) for layer in out_layers]

    anchor_infos = list(zip(layer_dims, anchor_scales_list))
    print(list(anchor_infos))
    print("1. network setup is done.")

    with tf.name_scope('raw_label'):
      _y = []
      for layer_dim, anchor_scale in anchor_infos:
        (h_num_grid, w_num_grid) = layer_dim
        box_dim = (FLAGS.nclass + 4)*len(anchor_scale)
        ph = tf.placeholder(tf.float32, [FLAGS.batch_size, h_num_grid, w_num_grid, box_dim])
        _y.append(ph)
    print("2. label setup is done.")

    with tf.name_scope('cal_loss'):
      loss = calculate_loss(_y, out_layers, anchor_scales_list)
      regularization_loss = tf.losses.get_regularization_loss(scope='(ssd|vgg)')
      total_loss = loss + regularization_loss

      tf.summary.scalar('total_loss', total_loss)
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('regularization_loss', regularization_loss)
    print("3. loss setup is done.")

    epoch_step, epoch_update = utils.get_epoch()
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print("==get_opt()============================")
    for item in var_list:
      print(item.name)
    with tf.name_scope('train'):
      opt, lr_decay_op1, lr_decay_op2 = get_opt(total_loss, '(ssd|vgg)')
    print("4. optimizer setup is done.")

    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    print("5. misc setup is done.")

    merged = tf.summary.merge_all()
    print("6. summary setup is  done.")

    start = datetime.now()
    print("Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S"))

    config=tf.ConfigProto()
    #config.log_device_placement=True
    config.intra_op_parallelism_threads=FLAGS.num_threads
    with tf.Session(config=config) as sess:
      init_vgg_16_fn(sess)

      sess.run(init_op)

      writer.add_graph(sess.graph)
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

      epoch_restored = sess.run(epoch_step)
      for epoch in range(epoch_restored, FLAGS.max_epoch):
        print("#####################################################################")
        print("epoch: {}".format(epoch))

        random.shuffle(filelist)
        max_itr = len(filelist)//FLAGS.batch_size

        img_save_dir = os.path.join(FLAGS.save_dir, "epoch_%03d"%epoch)
        if not os.path.exists(img_save_dir):
          os.mkdir(img_save_dir)

        for itr in range(0, max_itr):
          print("===================================================================")
          print("[{}] {}/{}".format(epoch, itr, max_itr))
          step = epoch*max_itr + itr

          # build minibatch
          b = itr*FLAGS.batch_size
          e = b + FLAGS.batch_size
          _batch = filelist[b:e]
          utils.debug_print(FLAGS.dbprint, _batch)

          feed_imgs = utils.load_imgs(FLAGS.train_img_dir, _batch)
          _feed_annots = utils.load_annots(FLAGS.train_annot_dir, _batch)

          feed_scaletrans, feed_flips, feed_annots_list = build_feed_annots(_feed_annots, anchor_infos)

          assert len(list(anchor_infos)) == len(feed_annots_list), "anchor_infos and feed_annots_list should have same length"

          feed_dict = {_x: feed_imgs, _st: feed_scaletrans, _flip: feed_flips, drop_prob:0.5}
#          print(_y)
          for ph, feed_annots in zip(_y, feed_annots_list):

#            print("------------------------------------------------------")
#            print(ph)
#            print(feed_annots.shape)
            feed_dict[ph] = feed_annots
#          test = tf.get_default_graph().get_tensor_by_name("ssd/backend0/weights:0")
#          test = tf.get_default_graph().get_tensor_by_name("ssd/backend0/biases:0")

#          print("test before:", test.eval())
#          var_grad = tf.gradients(loss, [test])[0]
#          var_grad_val = sess.run([var_grad], feed_dict=feed_dict)
#          print("test var_grad:", np.sum(var_grad_val))
#          print("test var_grad:", var_grad_val)
          summary, _, total_loss_val, loss_val, regularization_loss_val = sess.run([merged, opt, total_loss, loss, regularization_loss], feed_dict=feed_dict)
#          print("test after:", test.eval())

          print("total_loss: {}".format(total_loss_val))
          print("loss: {}, regularization_loss: {}".format(loss_val, regularization_loss_val))

          current = datetime.now()
          print('\telapsed:' + str(current - start))
          if itr % 100 == 0:
            #print("input filename:{}".format(_batch[0]))
            data_val, aug_val, label_val, out_val = sess.run([_x, aug, _y, out_layers], feed_dict=feed_dict)
            #label_val = sess.run(_y, feed_dict=feed_dict)
            #out_val = sess.run(out_layers, feed_dict=feed_dict)
            orig_img = cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR)
            # crop region
            cr = feed_scaletrans[0]*FLAGS.img_orig_size
            cr = cr.astype(np.int)
            orig_img = improc.visualization_orig(orig_img, _feed_annots[0], idx2obj, palette)
            orig_img = cv2.rectangle(orig_img, (cr[1], cr[0]), (cr[3], cr[2]), (255,255,255), 2)
            orig_img = cv2.resize(orig_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            orig_img = cv2.putText(orig_img, _batch[0], (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            aug_img = cv2.cvtColor(aug_val[0], cv2.COLOR_RGB2BGR)
            out_img = aug_img.copy()
            label_val = [label[0] for label in label_val]
            aug_img = cv2.resize(aug_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            aug_img = visualization(aug_img, label_val, anchor_infos, idx2obj, palette)

            out_val = [out[0] for out in out_val]
            out_img = cv2.resize(out_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            out_img = visualization(out_img, out_val, anchor_infos, idx2obj, palette, options=[])
            #cv2.imshow('input', improc.img_listup([orig_img, aug_img, out_img]))
            img_save_path = os.path.join(img_save_dir, 'result_%05d.png'%(itr))
            cv2.imwrite(img_save_path, improc.img_listup([orig_img, aug_img, out_img]))

          writer.add_summary(summary, step)
          key = cv2.waitKey(5)
          if key == 27:
            sys.exit()

            #compare(feed_annots_list[0], out_val[0])

        print("#######################################################")
        _ = sess.run(epoch_update)
        saver.save(sess, checkpoint)

  writer.close()

if __name__ == "__main__":
  tf.app.run()
