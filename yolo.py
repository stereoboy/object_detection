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
import utils
import common
import voc
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
tf.flags.DEFINE_float("learning_rate0", "1e-5", "initial learning rate")
tf.flags.DEFINE_float("learning_rate1", "1e-6", "learning rate for epoch 60")
tf.flags.DEFINE_float("learning_rate2", "1e-7", "learning rate for epoch 90")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_float("eps", "1e-5", "epsilon for various operation")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("pt_w", "0.1", "weight of pull-away term")
tf.flags.DEFINE_float("margin", "20", "Margin to converge to for discriminator")
tf.flags.DEFINE_string("noise_type", "uniform", "noise type for z vectors")
tf.flags.DEFINE_integer("channel", "3", "batch size for training")
tf.flags.DEFINE_integer("img_orig_size", "646", "sample image size")
tf.flags.DEFINE_integer("img_size", "448", "sample image size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("save_dir", "yolo_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")

def load_imgs(filelist):
  def load_img(path):
    _img = Image.open(path)
    img = np.array(_img)
    _img.close()
    return img

  _imgs = [os.path.join(FLAGS.train_img_dir, filename + ".png") for filename in filelist]

  imgs = [load_img(_img) for _img in _imgs]
  return imgs

def visualization(img, annot, palette, out=False):
  print("visualization()")
  h, w = img.shape[:2]

  num_grid = annot.shape[0]
  grid_size = FLAGS.img_size/num_grid
  vis_grid = np.zeros_like(img)

  for row in range(num_grid):
    for col in range(num_grid):
      cell_info_dim = FLAGS.nclass + FLAGS.B*(1 + 4)
      #confidence0 = annot[row, col, FLAGS.nclass] # confidence value of the first object

      idx = int(1 + np.argmax(annot[row, col, :FLAGS.nclass]))
      _color = palette[idx]
      color = (int(_color[2]), int(_color[1]), int(_color[0]))
      name = voc.idx2obj[idx]
      for k in range(FLAGS.B):
        b = FLAGS.nclass + k*(1 + 4)
        e = b + (1 + 4)
        c, cx, cy, nw, nh = annot[row, col, b:e]

        #c = np.max(annot[row, col, :FLAGS.nclass])
        if c > FLAGS.confidence:
          (y_loc, x_loc) = (row, col)

          name = name + '%2d'%(100*np.max(annot[row, col, :FLAGS.nclass])) + '_%.2f'%(c)
          cx = grid_size*(x_loc + cx)
          cy = grid_size*(y_loc + cy)
          bw = FLAGS.img_size*nw
          bh = FLAGS.img_size*nh

#          print("{} is located at ({}, {})".format(name, row, col))
#          print("WxH at {}x{}".format(bw, bh))

          grid_b = (int(grid_size*x_loc), int(grid_size*y_loc))
          grid_e = (int(grid_size*(x_loc+1)), int(grid_size*(y_loc+1)))

          b = (int(common.clip((cx - 0.5*bw), 0.0, FLAGS.img_size)), int(common.clip((cy - 0.5*bh), 0.0, FLAGS.img_size)))
          e = (int(common.clip((cx + 0.5*bw), 0.0, FLAGS.img_size)), int(common.clip((cy + 0.5*bh), 0.0, FLAGS.img_size)))

          #print(b, e)

          vis_grid= cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
          img = cv2.rectangle(img, b, e, color, 7)
          img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
          img = cv2.circle(img, (int(cx), int(cy)), 4, color, -1)
          #img = cv2.circle(img, (int(cx), int(cy)), 4, (255,255,255), -1)

  img = 0.7*img + 0.3*vis_grid
  return img

def compare(annot, out_annot):
  num_grid = annot.shape[0]
  grid_size = FLAGS.img_size/num_grid

  print(out_annot[:, :, FLAGS.nclass])
  pred_class = np.argmax(out_annot[:, :, :FLAGS.nclass], axis=2) + 1
  for i in range(num_grid):
    pred_name = []
    for j in range(num_grid):
      pred_name.append(voc.idx2obj[pred_class[i, j]])
    print(pred_name)

  for row in range(num_grid):
    for col in range(num_grid):
      idx = int(1 + np.argmax(annot[row, col, :FLAGS.nclass]))
      name = voc.idx2obj[idx]

      for k in range(FLAGS.B):
        b = FLAGS.nclass + k*(1 + 4)
        e = b + (1 + 4)
        c, cx, cy, nw, nh = annot[row, col, b:e]

        if c > FLAGS.confidence:
          #out_idx = int(1 + np.argmax(out_annot[row, col, :FLAGS.nclass]))
          out_idx = (np.argsort(out_annot[row, col, :FLAGS.nclass])[-5:][::-1] + 1)
          #out_name = voc.idx2obj[out_idx]
          out_name = [ voc.idx2obj[i] for i in out_idx]
          out_name_check = (out_idx == idx)
          out_c, out_cx, out_cy, out_nw, out_nh = out_annot[row, col, b:e]

          print("at ({}, {})".format(row, col))
          print("class: {} vs {}".format(name, out_name))
          print("class: {} vs {}".format(name, out_name_check))
          print("class: {} vs {}".format(name, np.sort(out_annot[row, col, :FLAGS.nclass])[-5:][::-1]))
          print("confidence:{} vs {}".format(c, out_c))
          print("x, y: ({}, {}) vs ({}, {})".format(cx, cy, out_cx, out_cy))
          print("w, h: {}x{} vs {}x{}".format(nw, nh, out_nw, out_nh))


def load_annots(filelist):
  def load_annot(path):
    #print(path)
    annot = np.load(path, encoding='bytes')
    #print("original dims: {}x{}".format(annot[0,0], annot[0,1]))
    return  annot

  _annots = [os.path.join(FLAGS.train_annot_dir, filename + ".npy") for filename in filelist]

  annots = [load_annot(_annot) for _annot in _annots]

  return annots

def build_feed_annots(_feed_annots):
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
      x1, x2 = np.max((0.0, x1)), np.min((w, x2))
      y1, y2 = np.max((0.0, y1)), np.min((h, y2))

      if feed_flips[i]:
        x1 = w - x1
        x2 = w - x2
        tmp = x1
        x1 = x2
        x2 = tmp

      idx = int(idx)
      (x_loc, y_loc), (cx, cy, nw, nh) = common.cal_rel_coord(w, h, x1, x2, y1, y2, w_grid, h_grid)

      if x_loc < 0 or y_loc < 0 or x_loc >= FLAGS.num_grid or y_loc >= FLAGS.num_grid:
        continue
      if nw < 0 or nh < 0:
        continue

      # if object is still on cropped region
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


#def random_crop(value, size, seed=None, name=None):
#  """Randomly crops a tensor to a given size.
#  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
#  Requires `value.shape >= size`.
#  If a dimension should not be cropped, pass the full size of that dimension.
#  For example, RGB images can be cropped with
#  `size = [crop_height, crop_width, 3]`.
#  Args:
#    value: Input tensor to crop.
#    size: 1-D tensor with size the rank of `value`.
#    seed: Python integer. Used to create a random seed. See
#      @{tf.set_random_seed}
#      for behavior.
#    name: A name for this operation (optional).
#  Returns:
#    A cropped tensor of the same rank as `value` and shape `size`.
#  """
#  # TODO(shlens): Implement edge case to guarantee output size dimensions.
#  # If size > value.shape, zero pad the result so that it always has shape
#  # exactly size.
#  with ops.name_scope(name, "random_crop", [value, size]) as name:
#    value = ops.convert_to_tensor(value, name="value")
#    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
#    shape = array_ops.shape(value)
#    check = control_flow_ops.Assert(
#        math_ops.reduce_all(shape >= size),
#        ["Need value.shape >= size, got ", shape, size])
#    shape = control_flow_ops.with_dependencies([check], shape)
#    limit = shape - size + 1
#    offset = random_uniform(
#        array_ops.shape(shape),
#        dtype=size.dtype,
#        maxval=size.dtype.max,
#        seed=seed) % limit
#    return array_ops.slice(value, offset, size, name=name)
#

def init_YOLOBE():
  def init_with_normal():

    return tf.contrib.layers.xavier_initializer()
    #return tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

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
      "17":tf.get_variable('e_bias_17', shape = [512], initializer=tf.zeros_initializer()),
      "18":tf.get_variable('e_bias_18', shape = [1024], initializer=tf.zeros_initializer()),

      "19":tf.get_variable('e_bias_19', shape = [512], initializer=tf.zeros_initializer()),
      "20":tf.get_variable('e_bias_20', shape = [1024], initializer=tf.zeros_initializer()),

      "21":tf.get_variable('e_bias_21', shape = [1024], initializer=tf.zeros_initializer()),
      "22":tf.get_variable('e_bias_22', shape = [1024], initializer=tf.zeros_initializer()),

      # step 6
      "23":tf.get_variable('e_bias_23', shape = [1024], initializer=tf.zeros_initializer()),
      "24":tf.get_variable('e_bias_24', shape = [1024], initializer=tf.zeros_initializer()),
      }

  WFCs = {
      "1":tf.get_variable('fc_1', shape = [FLAGS.num_grid*FLAGS.num_grid*1024, 512], initializer=init_with_normal()),
      "2":tf.get_variable('fc_2', shape = [512, 4096], initializer=init_with_normal()),
      "3":tf.get_variable('fc_3', shape = [4096, FLAGS.num_grid*FLAGS.num_grid*(FLAGS.nclass + 5*FLAGS.B)], initializer=init_with_normal()),
      }

  BFCs = {
      "1":tf.get_variable('fcb_1', shape = [512], initializer=tf.zeros_initializer()),
      "2":tf.get_variable('fcb_2', shape = [4096], initializer=tf.zeros_initializer()),
      "3":tf.get_variable('fcb_3', shape = [FLAGS.num_grid*FLAGS.num_grid*(FLAGS.nclass + 5*FLAGS.B)], initializer=tf.zeros_initializer()),
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
#  normalized = batch_norm_layer(biased, "bne" + name, reuse)
#  relued = leaky_relu(normalized)
  relued = leaky_relu(biased)

  return relued

def model_YOLO(x, WEs, BEs, WFCs, BFCs, drop_prob, reuse=False):

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
  #batch_size = relued.get_shape()[0]
  #fc = tf.reshape(relued, shape=[batch_size, -1]) # [batch_size, 7*7*1024]
  fc = tf.reshape(relued, shape=[-1, 7*7*1024])

  fc = tf.nn.bias_add(tf.matmul(fc, WFCs['1']), BFCs['1'])
  relued = leaky_relu(fc)

  fc = tf.nn.bias_add(tf.matmul(fc, WFCs['2']), BFCs['2'])
  relued = leaky_relu(fc)

  dropouted = tf.nn.dropout(relued, drop_prob)

  fc = tf.nn.bias_add(tf.matmul(dropouted, WFCs['3']), BFCs['3'])

  final = tf.reshape(fc, shape=[-1, (FLAGS.nclass + 5*FLAGS.B), 7, 7])

  # no activation or linear activation

  return final

def get_epoch():
  epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
  epoch_update = epoch_step.assign(epoch_step + 1)
  return epoch_step, epoch_update

def get_opt(loss, scope):

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

#  learning_rate = tf.train.exponential_decay(FLAGS.learning_rate0, global_step,
#                                                 1, 0.998, staircase=True)
  learning_rate = tf.Variable(FLAGS.learning_rate0, trainable=False)
  lr_decay_op1 = tf.assign(learning_rate, FLAGS.learning_rate1)
  lr_decay_op2 = tf.assign(learning_rate, FLAGS.learning_rate2)
  learning_rate = tf.Print(learning_rate, [learning_rate], message="learning_rate:")
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         FLAGS.momentum).minimize(loss,
                                                       var_list=var_list,
                                                       global_step=global_step)
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

  return optimizer, lr_decay_op1, lr_decay_op2
#  return tf.train.AdamOptimizer(0.0001).minimize(loss)
#  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
#  grads = optimizer.compute_gradients(loss, var_list=var_list)
#  return optimizer.apply_gradients(grads)

def calculate_loss(y, out):
  # devide y into each boundbox infos and class info
  yClass = y[:, :, :, :FLAGS.nclass]
  obj    = y[:, :, :, FLAGS.nclass:FLAGS.nclass + 1]
  yBBs   = y[:, :, :, FLAGS.nclass:FLAGS.nclass + 5]

  # devide output into each boundbox infos and class info
  Class = out[:, :, :, :FLAGS.nclass]
  BBs = out[:, :, :, FLAGS.nclass:]

  # calculate boundbox infos
  lambda_coord = 5
  lambda_noobj = 0.5

  coord_term  = 0
  confi_term  = 0

  yC, yXY, yWH = yBBs[:, :, :, 0:1], yBBs[:, :, :, 1:3], yBBs[:, :, :, 3:5]

  yArea = yWH[:,:,:,0]*yWH[:,:,:,1]
  yTopLeft = yXY - 0.5*yWH
  yBotRight = yXY + 0.5*yWH

  IOUs = []
  for i in range(FLAGS.B):
    offset = i*5
    Conf = BBs[:, :, :, offset:offset + 1]
    XY = BBs[:, :, :, offset + 1:offset +3]
    sqrtWH  = BBs[:, :, :, offset + 3:offset + 5]
    WH = tf.square(sqrtWH)

    Area = WH[:,:,:,0]*WH[:,:,:,1]
    TopLeft = XY - 0.5*WH
    BotRight = XY + 0.5*WH

    interTopLeft = tf.maximum(yTopLeft, TopLeft)
    interBotRight = tf.minimum(yBotRight, BotRight)
    interWH = interBotRight - interTopLeft
    interWH = tf.maximum(interWH, 0.0)
    iArea = interWH[:,:,:,0]*interWH[:,:,:,1]

    #uArea = tf.clip_by_value(yArea + Area - iArea, 1e-9, 1.0)
    uArea = yArea + Area - iArea
    iou = tf.expand_dims(tf.truediv(iArea, uArea), axis=3)
    IOUs.append(iou)

  IOUs = tf.concat(IOUs, axis=3)
  best_iou = tf.equal(IOUs, tf.reduce_max(IOUs, [3], True))
  best_iou = tf.cast(best_iou, tf.float32)
  Obj = best_iou*yC
  Score = Obj*IOUs

  #best_iou = tf.Print(best_iou, [best_iou], summarize=100, message="best_iou:")
  out_post = [Class]
  for i in range(FLAGS.B):
    offset = i*5
    C = BBs[:, :, :, offset:offset + 1]
    XY = BBs[:, :, :, offset + 1:offset +3]
    sqrtWH  = BBs[:, :, :, offset + 3:offset + 5]
    WH = tf.square(sqrtWH)

    #Score = best_iou[:, :, :, i:i + 1]*C
    ConfDiff = tf.square(Obj - C)

    t0 = lambda_coord*tf.reduce_sum(Obj*tf.square(yXY - XY), axis=[1,2,3])
#    t0 = tf.Print(t0, [t0, tf.shape(t0)[1:]], summarize=49, message="[{}] t0:".format(i))

    t1 = lambda_coord*tf.reduce_sum(Obj*tf.square(tf.sqrt(yWH) - sqrtWH), axis=[1,2,3])
#    t1 = tf.Print(t1, [t1, tf.shape(t1)[1:]], summarize=49, message="[{}] t1:".format(i))

    t2 = tf.reduce_sum(Obj*ConfDiff, axis=[1,2,3])
#    t2 = tf.Print(t2, [t2, tf.shape(t2)[1:]], summarize=49, message="[{}] t2".format(i))

    t3 = lambda_noobj*tf.reduce_sum((1 - Obj)*ConfDiff, axis=[1,2,3])
#    t3 = tf.Print(t3, [t3, tf.shape(t3)[1:]], summarize=49, message="[{}] t3:".format(i))

    coord_term += t0 + t1
    confi_term += t2 + t3
    #coord_term += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C, labels=yC))

    out_post.extend([Conf, XY, WH])

  out_post = tf.concat(out_post, axis=3)

  noobj = 1 - obj
  test = tf.squeeze(noobj[0])
  # calculate boundbox infos

#  sqrd_cls_err = tf.square(yClass - Class)
  #class_term = tf.reduce_sum(obj*sqrd_cls_err + lambda_noobj*noobj*sqrd_cls_err)
#  obj = tf.Print(obj, [obj, tf.shape(obj)[1:]], summarize=600, message="obj:")
#  sqrd_cls_err = tf.Print(sqrd_cls_err, [sqrd_cls_err, tf.shape(sqrd_cls_err)[1:]], summarize=600, message="sqrd_cls_err:")
  class_term = obj*tf.square(yClass - Class)
  class_term = tf.reduce_sum(class_term, axis=[1, 2, 3])

  coord_term = tf.Print(coord_term, [coord_term], summarize=10, message="coord_term:")
  confi_term = tf.Print(confi_term, [confi_term], summarize=10, message="confi_term:")
  class_term = tf.Print(class_term, [class_term], summarize=10, message="class_term:")
  loss = coord_term + confi_term + class_term
  #loss = class_term
  #loss = coord_term

  loss = tf.reduce_mean(loss)
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Class, labels=yClass))
  return loss, out_post, test

def evaluate(sess, filelist, x, y, out, accuracy):
  size = len(filelist)
  itr_size = size//FLAGS.batch_size
  total_accuracy_val = 0
  for itr in range(0, itr_size):
    # build minibatch
    _batch = filelist[itr:itr + FLAGS.batch_size]

    feed_imgs = load_imgs(_batch)
    _feed_annots = load_annots(_batch)

    feed_scaletrans, feed_flips, feed_annots = build_feed_annots(_feed_annots)
    accuracy_val = sess.run(accuracy, feed_dict=feed_dict)
    total_accuracy_val += accuracy_val

  return total_accuracy_val / itr_size

def main(args):

  colormap, palette = voc.build_colormap_lookup(21)
  idx2obj = voc.idx2obj
  cell_info_dim = FLAGS.nclass + FLAGS.B*(1 + 4) # 2x(confidence + (x, y, w, h)) + class

  #pool = mp.Pool(processes=6)

  drop_prob = tf.placeholder(tf.float32)
  _x = tf.placeholder(tf.float32, [None, FLAGS.img_orig_size, FLAGS.img_orig_size, FLAGS.channel])
  _y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_grid, FLAGS.num_grid, cell_info_dim])
  _st = tf.placeholder(tf.float32, [None, 4])
  _flip = tf.placeholder(tf.bool, [None])

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  pretrained = utils.load_pretrained("./VGG_16.npy")

  with open(FLAGS.filelist, "r") as f:
    filelist = json.load(f)

  img_list = [os.path.join(FLAGS.train_img_dir, filename + ".png") for filename in filelist]
  annot_list = [os.path.join(FLAGS.train_annot_dir, filename + ".label") for filename in filelist]

  mean = tf.constant(np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32))

  aug = improc.augment_scale_translate_flip(_x, FLAGS.img_size, _st, _flip, FLAGS.batch_size)
  aug = tf.map_fn(lambda x:improc.augment_br_sat_hue_cont(x), aug)
  x = tf.cast(aug, dtype=tf.float32) - mean
  x = improc.augment_gaussian_noise(x)
  y = _y

  #with tf.device(FLAGS.device):
  x = tf.transpose(x, perm=[0, 3, 1, 2])
  print("0. input setup is done.")


  with tf.variable_scope("vgg_16") as scope:
    Ws, Bs = vgg_16.init_VGG16(pretrained)

  with tf.variable_scope("YOLO") as scope:
    WEs, BEs, WFCs, BFCs, = init_YOLOBE()

  print("1. variable setup is done.")

  _out = vgg_16.model_VGG16(x, Ws, Bs)
  _out = model_YOLO(_out, WEs, BEs, WFCs, BFCs, drop_prob=drop_prob)
  print("2. model setup is done.")

  out = tf.transpose(_out, perm=[0, 2, 3, 1])
  loss, out_post, test = calculate_loss(y, out)
  print("3. loss setup is done.")

  epoch_step, epoch_update = get_epoch()
  opt, lr_decay_op1, lr_decay_op2 = get_opt(loss, "YOLO")
  print("4. optimizer setup is done.")

  init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

  print("all graph setup is done")

  start = datetime.now()
  print("Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S"))

  config=tf.ConfigProto()
  #config.log_device_placement=True
  config.intra_op_parallelism_threads=FLAGS.num_threads
  with tf.Session(config=config) as sess:
    sess.run(init_op)

#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess, coord)

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
        b = itr*FLAGS.batch_size
        e = b + FLAGS.batch_size
        _batch = filelist[b:e]

        feed_imgs = load_imgs(_batch)
        _feed_annots = load_annots(_batch)

        feed_scaletrans, feed_flips, feed_annots = build_feed_annots(_feed_annots)

        feed_dict = {_x: feed_imgs, _y: feed_annots, _st: feed_scaletrans, _flip: feed_flips, drop_prob:0.5}

        test = tf.get_default_graph().get_tensor_by_name("YOLO/e_conv_17:0")

        print("test before:", test.eval())
        var_grad = tf.gradients(loss, [test])[0]
        var_grad_val = sess.run([var_grad], feed_dict=feed_dict)
        print("test var_grad:", np.sum(var_grad_val))
        print("test var_grad:", var_grad_val)
        _, loss_val = sess.run([opt, loss, ], feed_dict=feed_dict)

        #print("test: {}".format(sess.run(test, feed_dict=feed_dict)))
        print("loss: {}".format(loss_val))
        current = datetime.now()
        print('\telapsed:' + str(current - start))

        print("test after:", test.eval())

        if itr % 1 == 0:
          data_val, aug_val, label_val, out_val = sess.run([_x, aug, _y, out_post], feed_dict=feed_dict)
          orig_img = cv2.cvtColor(data_val[0],cv2.COLOR_RGB2BGR)
          # crop region
          cr = feed_scaletrans[0]*FLAGS.img_orig_size
          cr = cr.astype(np.int)
          orig_img = improc.visualization_orig(orig_img, _feed_annots[0], idx2obj, palette)
          orig_img = cv2.rectangle(orig_img, (cr[1], cr[0]), (cr[3], cr[2]), (255,255,255), 2)
          orig_img = cv2.resize(orig_img, (FLAGS.img_size, FLAGS.img_size))

          aug_img = cv2.cvtColor(aug_val[0], cv2.COLOR_RGB2BGR)
          out_img = aug_img.copy()
          aug_img = visualization(aug_img, feed_annots[0], palette)

          out_img = visualization(out_img, out_val[0], palette, True)
          cv2.imshow('input', improc.img_listup([orig_img, aug_img, out_img]))

          compare(feed_annots[0], out_val[0])

        key = cv2.waitKey(0)
        if key == 27:
          sys.exit()
      print("#######################################################")
      _ = sess.run(epoch_update)
      saver.save(sess, checkpoint)


    cv2.destroyAllWindows()

if __name__ == "__main__":
  tf.app.run()
