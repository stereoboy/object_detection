import tensorflow as tf
import numpy as np
#import multiprocessing as mp
import glob
import os
import json
from datetime import datetime, date, time
import cv2
import sys
import common
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
tf.flags.DEFINE_integer("img_vis_size", "428", "sample image size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("save_dir", "yolov2_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")
tf.flags.DEFINE_float("weight_decay", "0.0005", "weight decay for L2 regularization")

slim = tf.contrib.slim

def visualization(img, annot, anchor_scales, idx2obj, palette):
  print("visualization()")
  h, w = [float(x) for x in img.shape[:2]]

  vis_grid = np.zeros_like(img)

  (h_num_grid, w_num_grid) = FLAGS.num_grid, FLAGS.num_grid

  for row in range(h_num_grid):
    for col in range(w_num_grid):
      offset = 0
      for i, anchor_scale in enumerate(anchor_scales):
        conf = annot[row, col, offset:offset + FLAGS.nclass]
        if np.max(conf) > .5:
          idx = int(1 + np.argmax(conf))
          _color = palette[idx]
          color = (int(_color[2]), int(_color[1]), int(_color[0]))
          name = idx2obj[idx]

          anchor_w, anchor_h = anchor_scale
          anchor_cx, anchor_cy = (col + .5), (row  + .5)

          b = offset + FLAGS.nclass
          e = b + 1 + 4
          iou, reg_cx, reg_cy, reg_nw, reg_nh = annot[row, col, b:e]

          cx, cy = reg_cx + anchor_cx, reg_cy + anchor_cy
          nw, nh = np.exp(reg_nw)*anchor_w, np.exp(reg_nh)*anchor_h

          cwh = ((cx, cy), (nw, nh))
          print(cwh)
          print(anchor_w, anchor_h)
          print(reg_cx, reg_cy, reg_nw, reg_nh)
          print('--------------------------------------')
          ((x1, y1), (x2, y2)) = improc.cvt_cwh2bbox(cwh)
          name = name + '_%d'%(i) + '_%.2f'%(np.max(conf))

#          print("{} is located at ({}, {})".format(name, row, col))
#          print("WxH at {}x{}".format(bw, bh))

          w_grid, h_grid = (w/w_num_grid, h/h_num_grid)
          b = (int(x1*w_grid), int(y1*h_grid))
          e = (int(x2*w_grid), int(y2*h_grid))

          #vis_grid= cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
          img = cv2.rectangle(img, b, e, color, 5)
          img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
          img = cv2.circle(img, (int(cx*w_grid), int(cy*h_grid)), 4, color, -1)

        offset += (FLAGS.nclass + 1 + 4)

  return img

def leaky_relu(tensor):
  return tf.maximum(tensor*0.1, tensor)

def base_conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format='NCHW', scope=None):

  out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, data_format=data_format, activation_fn=None, normalizer_fn=None, scope=scope)
  out = slim.batch_norm(out, activation_fn=leaky_relu, scope='bn_' + scope)

  return out

def get_output_from_vgg16(end_points):
  for ep in end_points:
    print(ep)
  vgg_out0, vgg_out1 = end_points['vgg_16/pool4'], end_points['vgg_16/pool5']
  return vgg_out0, vgg_out1

def build_feed_annots(_feed_annots, anchor_scales):
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

  (h_num_grid, w_num_grid) = FLAGS.num_grid, FLAGS.num_grid
  box_dim = (FLAGS.nclass + 1 + 4)*len(anchor_scales)

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

      bbox = ((x1*w_num_grid/w, y1*h_num_grid/h), (x2*w_num_grid/w, y2*h_num_grid/h))
      ((cx, cy), (nw, nh)) = improc.cvt_bbox2cwh(bbox)
      x_loc, y_loc = int(cx), int(cy)

      if x_loc < 0 or y_loc < 0 or x_loc >= FLAGS.num_grid or y_loc >= FLAGS.num_grid:
        continue

      idx = int(idx)

      offset = 0
      for anchor_scale in anchor_scales:
        anchor_w, anchor_h = anchor_scale
        anchor_cx, anchor_cy = (x_loc + .5), (y_loc + .5)
        anchor_cwh = ((anchor_cx, anchor_cy), (anchor_w, anchor_h))
        # iou with the current anchor region is > 0.5
        anchor_bbox = improc.cvt_cwh2bbox(anchor_cwh)
        iou = improc.cal_iou(bbox, anchor_bbox)

        #print(y_loc, x_loc, cx, cy, x1, x2, y1, y2)
        feed_annots[i, y_loc, x_loc, offset + idx - 1] = 1

        b = offset + FLAGS.nclass
        e = b + 1 + 4
        reg_cx, reg_cy = (cx - anchor_cx), (cy - anchor_cy)
        reg_nw, reg_nh = np.log(nw/anchor_w), np.log(nh/anchor_h)
        feed_annots[i, y_loc, x_loc, b:e] = np.array((iou, reg_cx, reg_cy, reg_nw, reg_nh), np.float32)
        if i == 0:
          cwh = ((cx, cy), (nw, nh))
          print(cwh)
          print((anchor_w, anchor_h), iou)
          print('--------------------------------------')

        offset += (FLAGS.nclass + 1 + 4)

  return feed_scaletrans, feed_flips, feed_annots

def model_yolov2_backend(_out0, _out1, anchor_scales):
  print(_out0)
  print(_out1)
  assert _out0.get_shape()[-1] == 2*FLAGS.num_grid, "_out0 should be 13*2=26"
  assert _out1.get_shape()[-1] == FLAGS.num_grid, "_out1 should be 13"

  # re-orgarnize from large image to small image
  # 2x2x1 -> 1x1x4
  def reorganize(x):
    num_inputs = x.get_shape().as_list()[1]
    num_outputs = 4*num_inputs
    print('reorganize(x) {}x{}'.format(num_inputs, num_outputs))

    kernel = np.zeros((2, 2, num_inputs, num_outputs), dtype=np.float32)
    for i in range(kernel.shape[0]):
      for j in range(kernel.shape[1]):
        for k in range(kernel.shape[2]):
          kernel[i, j, k, 4*k + 2*i + j ] = 1.0
    print('kernel', kernel.shape)
    print('kernel', kernel)
    kernel = tf.Variable(kernel, dtype=tf.float32, trainable=False)
    out = tf.nn.conv2d(x, kernel, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return out

  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                      biases_initializer=tf.zeros_initializer(),
                      data_format='NCHW'):

    out0 = base_conv2d(_out0, 512, [3, 3], scope='backend0')
    out0 = reorganize(out0)

    out1 = base_conv2d(_out1, 1024, [3, 3], scope='backend1')
    out1 = base_conv2d(out1, 1024, [3, 3], scope='backend2')

    # merge filter dim
    _out = tf.concat([out0, out1], axis=1)

    detect_dims = len(anchor_scales)*(FLAGS.nclass + 1 + 4)
    out = base_conv2d(_out, 1024, [3, 3], scope='detect0')
    out = base_conv2d(out, detect_dims, [1, 1], scope='detect1')

  return out

def calculate_loss(y, out, anchor_scales):

  # calculate boundbox infos
  lambda_coord = 5.0
  lambda_noobj = 1.0

  class_term  = 0
  coord_term  = 0
  confi_term  = 0

  out_post = []
  for i in range(anchor_scales):
    offset = i*(FLAGS.nclass + 1 + 4)
    yClass = y[:, :, :, offset:offset + FLAGS.nclass]
    Class  = out[:, :, :, offset:offset + FLAGS.nclass]

    offset += FLAGS.nclass
    yC      = y[:, :, :, offset:offset + 1]
    C       = out[:, :, :, offset:offset + 1]
    yXY     = y[:, :, :, offset + 1:offset +3]
    XY      = out[:, :, :, offset + 1:offset +3]
    yWH     = y[:, :, :, offset + 3:offset + 5]
    sqrtWH  = out[:, :, :, offset + 3:offset + 5]
    WH      = tf.square(sqrtWH)

    Area = WH[:,:,:,0]*WH[:,:,:,1]
    TopLeft = XY - 0.5*WH
    BotRight = XY + 0.5*WH

    interTopLeft = tf.maximum(yTopLeft, TopLeft)
    interBotRight = tf.minimum(yBotRight, BotRight)
    interWH = interBotRight - interTopLeft
    interWH = tf.maximum(interWH, 0.0)
    iArea = interWH[:,:,:,0]*interWH[:,:,:,1]

    out_post.extend([Class, C, XY, WH])

  loss = 0.0 #

  return loss

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

  # this is from yolo-voc.cfg in darknet source code
  # 1.0 means individual grid cell width
  #anchor_scales =  [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]
  anchor_scales =  [(1.3221, 1.73145)]

  with tf.Graph().as_default():
    mean = tf.constant(np.array((_R_MEAN, _G_MEAN, _B_MEAN), dtype=np.float32))

    detect_dims = len(anchor_scales)*(FLAGS.nclass + 1 + 4)
    _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel])
    _y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_grid, FLAGS.num_grid, detect_dims])
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


    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(vgg_16.checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    vgg_out0, vgg_out1 = get_output_from_vgg16(end_points)

    with tf.variable_scope('yolov2') as scope:
      _out = model_yolov2_backend(vgg_out0, vgg_out1, anchor_scales)
    print("1. network setup is done.")


    regularization_loss = tf.losses.get_regularization_loss(scope='yolov2')
    print('regularization_loss:', regularization_loss)
    #tf.losses.add_loss(regularization_loss)

    out = tf.transpose(_out, perm=[0, 2, 3, 1])
    loss = calculate_loss(_y, out)
    print(loss)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
    print('A:',total_loss)

    #total_loss += regularization_loss
    print('B:', total_loss)
    print("2. loss setup is done.")


    test_input = tf.range(start=0, limit=(4*4*2), dtype=tf.float32)
    test_input = tf.reshape(test_input, (1, 4, 4, 2))

    a = np.zeros((2, 2, 2, 8))
    for i in range(a.shape[0]):
      for j in range(a.shape[1]):
        for k in range(a.shape[2]):
          print(i, j, k)
          a[i, j, k, 4*k + 2*i + j ] = 1.0
          print(a)
    #a = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    #a = np.expand_dims(a, axis=2)
    print('a', a.shape)
    print('a', a)
    kernel = tf.Variable(a, dtype=tf.float32, trainable=False)
    conved = tf.nn.conv2d(test_input, kernel, strides=[1, 2, 2, 1], padding='SAME')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    epoch_step, epoch_update = utils.get_epoch()
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

#      print(tensor[0].eval().shape)
#      print(test_input.eval())
#      print(kernel.eval())
#      print(kernel.eval().shape)
#      print(conved.eval())
#      print(conved.eval().shape)

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

          feed_scaletrans, feed_flips, feed_annots = build_feed_annots(_feed_annots, anchor_scales)

          feed_dict = {_x: feed_imgs, _y: feed_annots, _st: feed_scaletrans, _flip: feed_flips}

          print(_y)

          #_, _ = sess.run([opt, total_loss], feed_dict=feed_dict)
          if itr % 1 == 0:
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
            aug_img = cv2.resize(aug_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            aug_img = visualization(aug_img, label_val[0], anchor_scales, idx2obj, palette)

            out_img = cv2.resize(out_img, (FLAGS.img_vis_size, FLAGS.img_vis_size))
            out_img = visualization(out_img, out_val[0], anchor_scales, idx2obj, palette)
            cv2.imshow('input', improc.img_listup([orig_img, aug_img, out_img]))

          key = cv2.waitKey(0)
          if key == 27:
            sys.exit()

            #compare(feed_annots_list[0], out_val[0])

        print("#######################################################")
        _ = sess.run(epoch_update)
        saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()
