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
tf.flags.DEFINE_float("learning_rate0", "1e-3", "initial learning rate")
tf.flags.DEFINE_float("learning_rate1", "1e-4", "learning rate for epoch 60")
tf.flags.DEFINE_float("learning_rate2", "1e-5", "learning rate for epoch 90")
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

def visualization(img, annot, anchor_scales, idx2obj, palette, options=['draw_anchor', 'target']):
  print("visualization()")
  h, w = [float(x) for x in img.shape[:2]]

  vis_grid = np.zeros_like(img)

  (h_num_grid, w_num_grid) = FLAGS.num_grid, FLAGS.num_grid

  for row in range(h_num_grid):
    for col in range(w_num_grid):
      offset = 0
      for i, anchor_scale in enumerate(anchor_scales):
        class_pred = annot[row, col, offset:offset + FLAGS.nclass]
        obj_conf = annot[row, col, offset + FLAGS.nclass]
        if np.max(class_pred) > .5:
          idx = int(1 + np.argmax(class_pred))
          _color = palette[idx]
          color = (int(_color[2]), int(_color[1]), int(_color[0]))
          name = idx2obj[idx]

          anchor_w, anchor_h = anchor_scale
          anchor_cx, anchor_cy = (col + .5), (row  + .5)
          anchor_cwh = ((anchor_cx, anchor_cy), (anchor_w, anchor_h))
          anchor_bbox = improc.cvt_cwh2bbox(anchor_cwh)

          b = offset + FLAGS.nclass
          e = b + 1 + 4
          iou, reg_cx, reg_cy, reg_nw, reg_nh = annot[row, col, b:e]

          cx, cy = col + reg_cx, row + reg_cy
          nw, nh = np.exp(reg_nw)*anchor_w, np.exp(reg_nh)*anchor_h

          cwh = ((cx, cy), (nw, nh))
          print(cwh)
          print(anchor_w, anchor_h)
          print(reg_cx, reg_cy, reg_nw, reg_nh)
          print('--------------------------------------')
          ((x1, y1), (x2, y2)) = improc.cvt_cwh2bbox(cwh)
          name = name + '_%d'%(i) + '_%.2f'%(np.max(class_pred))

#          print("{} is located at ({}, {})".format(name, row, col))
#          print("WxH at {}x{}".format(bw, bh))

          w_grid, h_grid = (w/w_num_grid, h/h_num_grid)
          b = (int(x1*w_grid), int(y1*h_grid))
          e = (int(x2*w_grid), int(y2*h_grid))

          anchor_b = (int(anchor_bbox[0][0]*w_grid), int(anchor_bbox[0][1]*h_grid))
          anchor_e = (int(anchor_bbox[1][0]*w_grid), int(anchor_bbox[1][1]*h_grid))

          #vis_grid= cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
          if 'draw_anchor' in options:
            img = cv2.rectangle(img, anchor_b, anchor_e, (0, 0, 255), 3)
          img = cv2.rectangle(img, b, e, color, 5)
          img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
          img = cv2.circle(img, (int(cx*w_grid), int(cy*h_grid)), 4, color, -1)

        offset += (FLAGS.nclass + 1 + 4)

  return img

def leaky_relu(tensor):
  return tf.maximum(tensor*0.1, tensor)

def base_conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format='NCHW', scope=None):

  out = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, data_format=data_format, activation_fn=None, normalizer_fn=None, scope=scope)
  out = slim.batch_norm(out, activation_fn=leaky_relu, scope=scope+'_bn')

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
      if nw < 0 or nh < 0:
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
        reg_cx, reg_cy = (cx - x_loc), (cy - y_loc)
        reg_nw, reg_nh = np.log(nw/anchor_w), np.log(nh/anchor_h)
        feed_annots[i, y_loc, x_loc, b:e] = np.array((1.0, reg_cx, reg_cy, reg_nw, reg_nh), np.float32)
        if i == 0:
          print(feed_annots[i, y_loc, x_loc, b:e])
          print(nw/anchor_w, nh/anchor_h)
          cwh = ((cx, cy), (nw, nh))
          print(cwh)
          print((anchor_w, anchor_h), iou)
          print('--------------------------------------')

        offset += (FLAGS.nclass + 1 + 4)

  return feed_scaletrans, feed_flips, feed_annots

def model_yolov2_backend(_out0, _out1, anchor_scales):
  print(_out0)
  print(_out1)
  assert _out0.get_shape()[2] == 2*FLAGS.num_grid, "_out0 should be 26x26, 13*2=26"
  assert _out0.get_shape()[3] == 2*FLAGS.num_grid, "_out0 should be 26x26, 13*2=26"
  assert _out1.get_shape()[2] == FLAGS.num_grid, "_out1 should be 13x13"
  assert _out1.get_shape()[3] == FLAGS.num_grid, "_out1 should be 13x13"

  # subpixel cnn layer
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
                      #biases_initializer=tf.zeros_initializer(),
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
  for i, anchor_scale in enumerate(anchor_scales):
    anchor_w, anchor_h = anchor_scale

    offset = i*(FLAGS.nclass + 1 + 4)
    yClass = y[:, :, :, offset:offset + FLAGS.nclass]
    Class  = out[:, :, :, offset:offset + FLAGS.nclass]

    offset += FLAGS.nclass
    Obj     = y[:, :, :, offset:offset + 1]
    Conf    = tf.sigmoid(out[:, :, :, offset:offset + 1])

    yXY     = y[:, :, :, offset + 1:offset +3]
    XY      = tf.sigmoid(out[:, :, :, offset + 1:offset +3])

    anchor_wh = tf.constant([anchor_w, anchor_h], dtype=tf.float32)
    yRegWH  = y[:, :, :, offset + 3:offset + 5]
    yWH     = anchor_wh*tf.exp(yRegWH)
    RegWH   = out[:, :, :, offset + 3:offset + 5]
    WH      = anchor_wh*tf.exp(RegWH)

    yArea = yWH[:,:,:,0]*yWH[:,:,:,1]
    yTopLeft = yXY - 0.5*yWH
    yBotRight = yXY + 0.5*yWH
    TopLeft = XY - 0.5*WH
    BotRight = XY + 0.5*WH
    Area = WH[:,:,:,0]*WH[:,:,:,1]

    interTopLeft = tf.maximum(yTopLeft, TopLeft)
    interBotRight = tf.minimum(yBotRight, BotRight)
    interWH = interBotRight - interTopLeft
    interWH = tf.maximum(interWH, 0.0)
    iArea = interWH[:,:,:,0]*interWH[:,:,:,1]
    uArea = yArea + Area - iArea
    iou = tf.expand_dims(tf.truediv(iArea, uArea), axis=3)

    ConfDiff   = tf.square(Obj*iou - Conf)
    t0 = lambda_coord*tf.reduce_sum(Obj*tf.square(yXY - XY), axis=[1,2,3])
    t1 = lambda_coord*tf.reduce_sum(Obj*tf.square(yRegWH - RegWH), axis=[1,2,3])

    t2 = tf.reduce_sum(Obj*ConfDiff, axis=[1,2,3])
    t3 = lambda_noobj*tf.reduce_sum((1 - Obj)*ConfDiff, axis=[1,2,3])

    coord_term += t0 + t1
    confi_term += t2 + t3

    class_term = Obj*tf.square(yClass - Class)
    class_term = tf.reduce_sum(class_term, axis=[1, 2, 3])

    coord_term = tf.Print(coord_term, [coord_term], summarize=10, message="coord_term:")
    confi_term = tf.Print(confi_term, [confi_term], summarize=10, message="confi_term:")
    class_term = tf.Print(class_term, [class_term], summarize=10, message="class_term:")

    out_post.extend([Class, Conf, XY, RegWH])
  out_post = tf.concat(out_post, axis=3)

  loss = coord_term + confi_term + class_term
  loss = tf.reduce_mean(loss)

  return loss, out_post

def get_opt(loss, scope):

  print('loss:{}'.format(loss))

  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print("==get_opt()============================")
  print(scope)
  for item in var_list:
    print(item.name)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.Variable(FLAGS.learning_rate0, trainable=False)
  lr_decay_op1 = tf.assign(learning_rate, FLAGS.learning_rate1)
  lr_decay_op2 = tf.assign(learning_rate, FLAGS.learning_rate2)
  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")

#  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  opt = optimizer.minimize(loss, var_list=var_list, global_step=global_step)

  return opt, lr_decay_op1, lr_decay_op2

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
  anchor_scales =  [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]
  #anchor_scales =  [(1.3221, 1.73145)]

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

    out = tf.transpose(_out, perm=[0, 2, 3, 1])
    loss, out_post = calculate_loss(_y, out, anchor_scales)
    regularization_loss = tf.losses.get_regularization_loss(scope='yolov2')
    total_loss = loss + regularization_loss

    #total_loss += regularization_loss
    print("2. loss setup is done.")

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print("==get_opt()============================")
    for item in var_list:
      print(item.name)
    opt, lr_decay_op1, lr_decay_op2 = get_opt(total_loss, 'yolov2')
    print("3. optimizer setup is done.")

    epoch_step, epoch_update = utils.get_epoch()
    init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
    print("4. misc setup is done.")


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
          b = itr*FLAGS.batch_size
          e = b + FLAGS.batch_size
          _batch = filelist[b:e]

          feed_imgs = utils.load_imgs(FLAGS.train_img_dir, _batch)
          _feed_annots = utils.load_annots(FLAGS.train_annot_dir, _batch)

          feed_scaletrans, feed_flips, feed_annots = build_feed_annots(_feed_annots, anchor_scales)

          feed_dict = {_x: feed_imgs, _y: feed_annots, _st: feed_scaletrans, _flip: feed_flips}

          _, total_loss_val, loss_val, regularization_loss_val = sess.run([opt, total_loss, loss, regularization_loss], feed_dict=feed_dict)

          print("total_loss: {}".format(total_loss_val))
          print("loss: {}, regularization_loss: {}".format(loss_val, regularization_loss_val))

          if itr % 5 == 0:
            data_val, aug_val, label_val, out_val = sess.run([_x, aug, _y, out_post], feed_dict=feed_dict)
#            label_val = sess.run(_y, feed_dict=feed_dict)
#            out_val = sess.run(out_post, feed_dict=feed_dict)
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
            out_img = visualization(out_img, out_val[0], anchor_scales, idx2obj, palette, options=[])
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
