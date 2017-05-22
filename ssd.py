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
import voc 
import utils
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
tf.flags.DEFINE_integer("B", "2", "number of Bound Box in grid cell")
tf.flags.DEFINE_integer("num_grid", "7", "number of grids vertically, horizontally")
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
tf.flags.DEFINE_integer("img_size", "300", "sample image size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("save_dir", "ssd_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")
tf.flags.DEFINE_float("weight_decay", "0.0005", "weight decay for L2 regularization")

slim = tf.contrib.slim

def build_feed_annots(_feed_annots, out_layers):
  batch_size = len(_feed_annots)

  cell_info_dim = FLAGS.nclass + FLAGS.B*(1 + 4)
  feed_annots = np.zeros((batch_size, FLAGS.num_grid, FLAGS.num_grid, cell_info_dim), np.float32)

  # scale and translation
  # input image is resized 537x537
  # we will choose random crop size and offset, and resize into 428x428
  # This is equivalent to resize ~20% and crop with the widnow of 428x428
  scales = np.random.uniform(0.60, 1.0, [batch_size, 1])
  offsets = (1 - scales)*np.random.uniform(0.0, 1.0, [batch_size, 2])
  ends = offsets + scales
  feed_scaletrans = np.concatenate([offsets, ends], axis=1)
  feed_flips = np.random.randint(0, 2, [batch_size])

  # build augmented annotations
  for i, _annot in enumerate(_feed_annots):
    # each image
    annot = _annot.copy()
    _w, _h = _annot[0, :2]
    scale = scales[i, 0] # w, h scale is same
    w, h = (_w*scale, _h*scale)
    w_grid, h_grid = (w/FLAGS.num_grid, h/FLAGS.num_grid)
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
    _annot_data = {}
    for box in annot[1:]:
      idx, x1, x2, y1, y2 = box
      x1 = max(0.0, x1)
      x2 = min(w, x2)
      y1 = max(0.0, y1)
      y2 = min(h, y2)

      if feed_flips[i]:
        x1 = w - x1
        x2 = w - x2
        tmp = x1
        x1 = x2
        x2 = tmp

      idx = int(idx)
      (x_loc, y_loc), (cx, cy, nw, nh) = voc.cal_rel_coord(w, h, x1, x2, y1, y2, w_grid, h_grid)

      # if object is still on cropped region
      if x_loc >= 0 and x_loc < 7 and y_loc >= 0 and y_loc < 7:
        if not (x_loc, y_loc) in _annot_data.keys():
          _annot_data[(x_loc, y_loc)] = (idx, [])
          _annot_data[(x_loc, y_loc)][1].append((1.0, cx, cy, nw, nh))
        elif _annot_data[(x_loc, y_loc)][0] == idx:
          _annot_data[(x_loc, y_loc)][1].append((1.0, cx, cy, nw, nh))

    for (x_loc, y_loc), (idx, bbs)  in _annot_data.items():
      #print (x_loc, y_loc, idx, bbs)
      feed_annots[i, y_loc, x_loc, idx-1] = 1
      #for bbi in range(min(2, len(bbs))):
      bbi =  np.random.randint(0, len(bbs))
      #for bbi in range(1):
      b = FLAGS.nclass
      e = b + (1 + 4)
      feed_annots[i, y_loc, x_loc, b:e] = np.array(bbs[bbi], np.float32)

  # annot
  return feed_scaletrans, feed_flips, feed_annots


def init_scales(num_layers):

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
    if 2 <= k <= 4:
      scales.append((s_k*np.sqrt(3.0), s_k*np.sqrt(1.0/3.0)))
      scales.append((s_k*np.sqrt(1.0/3.0), s_k*np.sqrt(3.0)))
    scale_list.append(scales)
  #scales = [ min_scale + (max_scale - min_scale)/(num_layers - 1)*(k - 1) for k in range(1, num_layers + 1)]
  return scale_list

def model_ssd(frontend, ):

  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                      biases_initializer=tf.zeros_initializer(),
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
    conv9_2 = slim.conv2d(conv9_1, 512, [3, 3], stride=2, padding='SAME', scope='conv9_2')

  return conv6_2, conv7_2, conv8_2, conv9_2

def main(args):

  colormap, palette = voc.build_colormap_lookup(21)
  idx2obj = voc.idx2obj

  with open(FLAGS.filelist, "r") as f:
    filelist = json.load(f)

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  vgg_16.setup_vgg_16()

  with tf.Graph().as_default(): 
    mean = tf.constant(np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32))

    drop_prob = tf.placeholder(tf.float32)
    _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel])
    _st = tf.placeholder(tf.float32, [None, 4])
    _flip = tf.placeholder(tf.bool, [None])

    aug =improc.augment_scale_translate_flip(_x, FLAGS.img_size, _st, _flip, FLAGS.batch_size)
    aug = tf.map_fn(lambda x:improc.augment_br_sat_hue_cont(x), aug)
    x = tf.cast(aug, dtype=tf.float32) - mean
    x = improc.augment_gaussian_noise(x)

    x = tf.transpose(x, perm=[0, 3, 1, 2])

    with slim.arg_scope(vgg_16.vgg_arg_scope()):
      _, end_points = vgg_16.vgg_16_base(x)

    out_layers = []
    out_layers.append(end_points['vgg_16/conv4/conv4_3'])

    vgg_outs = end_points['vgg_16/conv5/conv5_3']
    out_layers.append(end_points['vgg_16/conv5/conv5_3'])

    print('vgg_outs', vgg_outs, vgg_outs.get_shape())
    print('end_points', end_points)
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(vgg_16.checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    with tf.variable_scope('ssd') as scope:
      layers = model_ssd(vgg_outs)

    out_layers.extend(layers)
    print('regularization:', tf.losses.get_regularization_losses(scope='ssd'))
    print('layers', layers)

    epoch_step, epoch_update = utils.get_epoch()
    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    print("==get_opt()============================")
    for item in var_list:
      print(item.name)

    tensor = tf.contrib.framework.get_variables_by_name('vgg_16/conv1/conv1_1/weights')
    print('....:', tensor[0].get_shape())

    print(out_layers)
    scales = init_scales(len(out_layers))
    print(scales)
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
      print(tensor[0].eval().shape)

      sess.run(init_op)
      for epoch in range(FLAGS.max_epoch):
        print("#####################################################################")
        epoch_val = sess.run(epoch_step)
        print("epoch: {}".format(epoch_val))

        random.shuffle(filelist)
        max_itr = len(filelist)//FLAGS.batch_size
        for itr in range(0, len(filelist)//FLAGS.batch_size):
        print("===================================================================")
        print("[{}] {}/{}".format(epoch_val, itr, max_itr))

        # build minibatch
        _batch = filelist[itr:itr + FLAGS.batch_size]

        feed_imgs = utils.load_imgs(_batch)
        _feed_annots = utils.load_annots(_batch)

        feed_scaletrans, feed_flips, feed_annots = build_feed_annots(_feed_annots, out_layers)

        if itr % 10 == 0:
          data_val, aug_val, label_val, out_val = sess.run([_x, aug, _y, out_post], feed_dict=feed_dict)
          orig_img = cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR)
          # crop region
          cr = feed_scaletrans[0]*FLAGS.img_orig_size
          cr = cr.astype(np.int)
          orig_img = imgproc.visualization_orig(orig_img, _feed_annots[0], FLAGS.num_grid, idx2obj, palette)
          orig_img = cv2.rectangle(orig_img, (cr[1], cr[0]), (cr[3], cr[2]), (255,255,255), 2)
          orig_img = cv2.resize(orig_img, (FLAGS.img_size, FLAGS.img_size))

          aug_img = cv2.cvtColor(aug_val[0], cv2.COLOR_RGB2BGR)
          out_img = aug_img.copy()
          aug_img = visualization2(aug_img, feed_annots[0], palette)

          out_img = visualization2(out_img, out_val[0], palette, True)
          cv2.imshow('input', imgproc.img_listup([orig_img, aug_img, out_img]))

          compare(feed_annots[0], out_val[0])

      print("#######################################################")
      _ = sess.run(epoch_update)
      saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()
