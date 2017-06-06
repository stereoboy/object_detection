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
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_float("confidence", "0.1", "confidence limit")
tf.flags.DEFINE_float("negative_ratio", "3.0", "ratio between negative and positive samples")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
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
tf.flags.DEFINE_string("save_dir", "ssd_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")
tf.flags.DEFINE_float("weight_decay", "0.0005", "weight decay for L2 regularization")

slim = tf.contrib.slim

def visualization(img, annots, anchor_infos, idx2obj, palette):
  print("visualization()")
  h, w = img.shape[:2]

  vis_grid = np.zeros_like(img)

  # iterate layers
  for i, (layer, anchor_scales) in enumerate(anchor_infos):
    (h_num_grid, w_num_grid) = int(layer.get_shape()[1]), int(layer.get_shape()[2])
    box_dim = (FLAGS.nclass + 4)*len(anchor_scales)

    annot = annots[i]
    for row in range(h_num_grid):
      for col in range(w_num_grid):
        offset = 0
        for j, anchor_scale in enumerate(anchor_scales):
          conf = annot[row, col, offset:offset + FLAGS.nclass]
          if np.max(conf) > .5:
            idx = int(1 + np.argmax(conf))
            _color = palette[idx]
            color = (int(_color[2]), int(_color[1]), int(_color[0]))
            name = idx2obj[idx]

            anchor_w, anchor_h = anchor_scale
            anchor_cx, anchor_cy = (col + .5)/w_num_grid, (row  + .5)/h_num_grid

            b = offset + FLAGS.nclass
            e = b + 4
            reg_cx, reg_cy, reg_nw, reg_nh = annot[row, col, b:e]

            cx, cy = reg_cx*anchor_w + anchor_cx, reg_cy*anchor_h + anchor_cy
            nw, nh = np.exp(reg_nw)*anchor_w, np.exp(reg_nh)*anchor_h

            cwh = ((cx, cy), (nw, nh))
            print(cwh)
            print(anchor_w, anchor_h)
            print(reg_cx, reg_cy, reg_nw, reg_nh)
            print('--------------------------------------')
            ((x1, y1), (x2, y2)) = improc.cvt_cwh2bbox(cwh)
            name = name + '%d_%d'%(i, j) + '_%.2f'%(np.max(conf))

#          print("{} is located at ({}, {})".format(name, row, col))
#          print("WxH at {}x{}".format(bw, bh))

            b = (int(x1*w), int(y1*h))
            e = (int(x2*w), int(y2*h))

            #vis_grid= cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
            img = cv2.rectangle(img, b, e, color, 7)
            img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            img = cv2.circle(img, (int(cx*w), int(cy*h)), 4, color, -1)

          offset += (FLAGS.nclass + 4)

  return img

def build_feed_annots(_feed_annots, anchor_infos):
  batch_size = len(_feed_annots)

  # scale and translation
  # input image is resized 537x537
  # we will choose random crop size and offset, and resize into 428x428
  # This is equivalent to resize ~20% and crop with the widnow of 428x428
  scales = np.random.uniform(0.70, 1.0, [batch_size, 1])
  offsets = (1 - scales)*np.random.uniform(0.0, 1.0, [batch_size, 2])
  ends = offsets + scales
  feed_scaletrans = np.concatenate([offsets, ends], axis=1)
  feed_flips = np.random.randint(0, 2, [batch_size])

  feed_annots_list = []

  # iterate layers
  for layer, anchor_scales in anchor_infos:
    (h_num_grid, w_num_grid) = int(layer.get_shape()[1]), int(layer.get_shape()[2])
    box_dim = (FLAGS.nclass + 4)*len(anchor_scales)

    feed_annots = np.zeros((batch_size, h_num_grid, w_num_grid, box_dim), np.float32)

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

        if x_loc < 0 or y_loc < 0 or x_loc >= FLAGS.num_grid or y_loc >= FLAGS.num_grid:
          continue

        offset = 0
        for anchor_scale in anchor_scales:
          anchor_w, anchor_h = anchor_scale
          anchor_cx, anchor_cy = (x_loc + .5)/w_num_grid, (y_loc + .5)/h_num_grid
          anchor_cwh = ((anchor_cx, anchor_cy), (anchor_w, anchor_h))
          # iou with the current anchor region is > 0.5
          anchor_bbox = improc.cvt_cwh2bbox(anchor_cwh)
          iou = improc.cal_iou(bbox, anchor_bbox)
          if iou > 0.5:
            feed_annots[i, y_loc, x_loc, offset + idx-1] = 1

            b = offset + FLAGS.nclass
            e = b + 4
            reg_cx, reg_cy = (cx - anchor_cx)/anchor_w, (cy - anchor_cy)/anchor_h
            reg_nw, reg_nh = np.log(nw/anchor_w), np.log(nh/anchor_h)
            feed_annots[i, y_loc, x_loc, b:e] = np.array((reg_cx, reg_cy, reg_nw, reg_nh), np.float32)
            if i == 0:
              print((anchor_w, anchor_h), iou)
              print('--------------------------------------')

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

def model_ssd(frontend):

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

def model_backend(out_layers, anchor_scales_list):

  ret_layers = []
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

    offset = 0
    print("i:",i, y, out)
    for j, anchor_scale in enumerate(anchor_scales):
      print("j:",j)
      class_y = y[:, :, :, offset:offset + FLAGS.nclass]
      print(class_y)
      class_y = tf.reshape(class_y, shape=[FLAGS.batch_size, -1, FLAGS.nclass])
      print(class_y)

      class_out = out[:, :, :, offset:offset + FLAGS.nclass]
      class_out = tf.reshape(class_out, shape=[FLAGS.batch_size, -1, FLAGS.nclass])

      coord_y = y[:, :, :, offset + FLAGS.nclass: offset + FLAGS.nclass + 4]
      coord_y = tf.reshape(coord_y, shape=[FLAGS.batch_size, -1, 4])

      coord_out = out[:, :, :, offset + FLAGS.nclass: offset + FLAGS.nclass + 4]
      coord_out = tf.reshape(coord_out, shape=[FLAGS.batch_size, -1, 4])

      flat_class_y.append(class_y)
      flat_class_out.append(class_out)
      flat_coord_y.append(coord_y)
      flat_coord_out.append(coord_out)

      offset += (FLAGS.nclass + 4)

  flat_class_y = tf.concat(flat_class_y, axis=1)
  flat_class_out = tf.concat(flat_class_out, axis=1)
  flat_coord_y = tf.concat(flat_coord_y, axis=1)
  flat_coord_out = tf.concat(flat_coord_out, axis=1)
  print('flat_class_y', flat_class_y)
  print('flat_class_out', flat_class_out)
  print('flat_coord_y', flat_coord_y)
  print('flat_coord_out', flat_coord_out)

  positive_mask = tf.reduce_max(flat_class_y, axis=2) > 0.5
  negative_mask = tf.logical_not(positive_mask)

  positive_mask = tf.cast(positive_mask, dtype=tf.float32)
  positive_num  = tf.reduce_sum(positive_mask, axis=1)
  negative_mask = tf.cast(negative_mask, dtype=tf.float32)
  negative_num  = tf.reduce_sum(negative_mask, axis=1)

  print('positive_mask', positive_mask)
  print('negative_mask', negative_mask)
  print('positive_num', positive_num)
  print('negative_num', negative_num)

  negative_num = tf.cast(tf.minimum(negative_num, positive_num*FLAGS.negative_ratio), dtype=tf.int32)

  conf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_class_out, labels=flat_class_y)

  negative_loss = 0
  for i in range(FLAGS.batch_size):
    values, indices = tf.nn.top_k(-(conf_loss[i]*negative_mask[i]), k=negative_num[i])
    negative_loss += tf.reduce_sum(values)

  negative_loss /= FLAGS.batch_size

  loc_loss = smooth_l1_loss(flat_coord_out - flat_coord_y)

  loss = tf.reduce_sum(conf_loss*positive_mask, axis=1)
  #loss += tf.reduce_sum(values, axis=1)
  loss += tf.reduce_sum(loc_loss, axis=[1, 2])
  print(loss)

  loss = tf.div(loss, positive_num)
  print(loss)
  loss = tf.reduce_mean(loss) + negative_loss
  print(loss)
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
  learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
#  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")
  lr_decay_op1 = learning_rate.assign(1e-4)
  lr_decay_op2 = learning_rate.assign(1e-5)
  optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
#  optimizer = tf.train.AdamOptimizer(learning_rate)
  opt = optimizer.minimize(loss, var_list=var_list, global_step=global_step)

#  learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
#  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")
#  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
#
#  learning_rate = tf.Variable(
#      float(1e-3), trainable=False, dtype=tf.float32)
#  lr_decay_op1 = learning_rate.assign(1e-3)
#  lr_decay_op2 = learning_rate.assign(1e-4)
#  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")

#  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
#                                                 10, 0.9995, staircase=True)
#  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
#                                                                var_list=var_list,
#                                                                global_step=global_step)

  return opt, lr_decay_op1, lr_decay_op2
#  return tf.train.AdamOptimizer(0.0001).minimize(loss)
#  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
#  grads = optimizer.compute_gradients(loss, var_list=var_list)
#  return optimizer.apply_gradients(grads)

def main(args):

  colormap, palette = voc.build_colormap_lookup(21)
  idx2obj = voc.idx2obj

  with open(FLAGS.filelist, "r") as f:
    filelist = json.load(f)

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  vgg_16.setup_vgg_16()

  _R_MEAN = 123.68
  _G_MEAN = 116.78
  _B_MEAN = 103.94

  with tf.Graph().as_default(): 
    mean = tf.constant(np.array((_R_MEAN, _G_MEAN, _B_MEAN), dtype=np.float32))

    drop_prob = tf.placeholder(tf.float32)
    _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel])
    _st = tf.placeholder(tf.float32, [None, 4])
    _flip = tf.placeholder(tf.bool, [None])

    aug = improc.augment_scale_translate_flip(_x, FLAGS.img_size, _st, _flip, FLAGS.batch_size)
    aug = tf.map_fn(lambda x:improc.augment_br_sat_hue_cont(x), aug)
    x = tf.cast(aug, dtype=tf.float32) - mean
    x = improc.augment_gaussian_noise(x)

    x = tf.transpose(x, perm=[0, 3, 1, 2])

    print("0. input image setup is done.")

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


    anchor_scales_list = init_anchor_scales(len(out_layers))
    with tf.variable_scope('ssd') as scope:
      out_layers = model_backend(out_layers, anchor_scales_list)

    anchor_infos = list(zip(out_layers, anchor_scales_list))
    print(list(anchor_infos))
    print("1. network setup is done.")

    print('regularization:', tf.losses.get_regularization_loss(scope='ssd'))
    print('layers', layers)

    regularization_loss = tf.losses.get_regularization_loss(scope='ssd')
    tf.losses.add_loss(regularization_loss)

    _y = []
    for layer, anchor_scale in anchor_infos:
      (h_num_grid, w_num_grid) = layer.get_shape()[1:3]
      box_dim = (FLAGS.nclass + 4)*len(anchor_scale)
      ph = tf.placeholder(tf.float32, [FLAGS.batch_size, h_num_grid, w_num_grid, box_dim])
      _y.append(ph)
    print("2. label setup is done.")

    loss = calculate_loss(_y, out_layers, anchor_scales_list)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()
    print('get_loss', tf.losses.get_losses())
    print("3. loss setup is done.")

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print("==get_opt()============================")
    for item in var_list:
      print(item.name)
    opt = get_opt(total_loss, 'ssd')
    print("4. optimizer setup is done.")

    epoch_step, epoch_update = utils.get_epoch()
    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

    print("5. misc setup is done.")

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
      for epoch in range(FLAGS.max_epoch):
        print("#####################################################################")
        epoch_val = sess.run(epoch_step)
        print("epoch: {}".format(epoch_val))

        #random.shuffle(filelist)
        max_itr = len(filelist)//FLAGS.batch_size
        for itr in range(0, len(filelist)//FLAGS.batch_size):
          print("===================================================================")
          print("[{}] {}/{}".format(epoch_val, itr, max_itr))

          # build minibatch
          _batch = filelist[itr:itr + FLAGS.batch_size]

          feed_imgs = utils.load_imgs(FLAGS.train_img_dir, _batch)
          _feed_annots = utils.load_annots(FLAGS.train_annot_dir, _batch)

          feed_scaletrans, feed_flips, feed_annots_list = build_feed_annots(_feed_annots, anchor_infos)

          assert len(list(anchor_infos)) == len(feed_annots_list), "anchor_infos and feed_annots_list should have same length"

          feed_dict = {_x: feed_imgs, _st: feed_scaletrans, _flip: feed_flips, drop_prob:0.5}
          print(_y)
          print("------------------------------------------------------")
          for ph, feed_annots in zip(_y, feed_annots_list):

            print(ph)
            print(feed_annots.shape)
            feed_dict[ph] = feed_annots

          print("------------------------------------------------------")

          _, _ = sess.run([opt, total_loss], feed_dict=feed_dict)
          if itr % 100 == 0:
            data_val, aug_val = sess.run([_x, aug], feed_dict=feed_dict)
            label_val = sess.run(_y, feed_dict=feed_dict)
            out_val = sess.run(_y, feed_dict=feed_dict)
            orig_img = cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR)
            # crop region
            cr = feed_scaletrans[0]*FLAGS.img_orig_size
            cr = cr.astype(np.int)
            orig_img = improc.visualization_orig(orig_img, _feed_annots[0], idx2obj, palette)
            orig_img = cv2.rectangle(orig_img, (cr[1], cr[0]), (cr[3], cr[2]), (255,255,255), 2)
            orig_img = cv2.resize(orig_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))

            aug_img = cv2.cvtColor(aug_val[0], cv2.COLOR_RGB2BGR)
            out_img = aug_img.copy()
            label_val = [label[0] for label in label_val]
            aug_img = cv2.resize(aug_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            aug_img = visualization(aug_img, label_val, anchor_infos, idx2obj, palette)

            out_val = [out[0] for out in out_val]
            out_img = cv2.resize(out_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            out_img = visualization(out_img, out_val, anchor_infos, idx2obj, palette)
            cv2.imshow('input', improc.img_listup([orig_img, aug_img, out_img]))

          key = cv2.waitKey(5)
          if key == 27:
            sys.exit()

            #compare(feed_annots_list[0], out_val[0])

        print("#######################################################")
        _ = sess.run(epoch_update)
        saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()
