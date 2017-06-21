import tensorflow as tf
import numpy as np
import cv2

def augment_br_sat_hue_cont(image):

  def _augment_br_sat_hue_cont(x):
    # all functions include clamping for overflow values
    x = tf.image.random_brightness(x, max_delta=0.3)
    x = tf.image.random_saturation(x, lower=0.7, upper=1.3)
    #x = tf.image.random_hue(x, max_delta=0.032)
    #x = tf.image.random_contrast(x, lower=0.7, upper=1.3)
    return x

  with tf.name_scope('random_sat_hue_cont'):
    image = tf.map_fn(lambda x:_augment_br_sat_hue_cont(x), image)
  return image

def augment_gaussian_noise(images, std=0.2):
  with tf.name_scope('gaussian_noise'):
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=std, dtype=tf.float32)
  return images + noise

def augment_scale_translate_flip(images, img_size, boxes, flip, batch_size, scale_range=0.2):

  #batch_size = images.get_shape()[0]
  #batch_size = FLAGS.batch_size # this value should be fixed up

  # Translation
  with tf.name_scope('scale_trans'):
    scale = 1.0 + tf.random_uniform([1], minval=0.0, maxval=scale_range)
    size = tf.constant([img_size, img_size])
    new_size = scale*tf.cast(size, dtype=tf.float32)

    box_ind = tf.range(start=0, limit=batch_size, dtype=tf.int32)

    images = tf.image.crop_and_resize(
        images,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=size
        )

#  def flip_left_right(i):
#    image = images[i]
#    flip_or_not = flip[i]
#    return tf.cond(flip_or_not, lambda: tf.reverse(image, axis=[1]), lambda: image)
#
#  with tf.name_scope('flip'):
#    idxs = tf.range(0, batch_size, dtype=tf.int32)
#    images = tf.map_fn(lambda idx:flip_left_right(idx), idxs, dtype=tf.float32)

  with tf.name_scope('flip'):
    flipeds = tf.reverse(images, axis=[2])
    images = tf.where(flip, flipeds, images)
  return images

def cal_area(box):
  w, h = np.array(box[1]) - np.array(box[0])
  return w*h

def cal_iou(box1, box2):
  area1 = cal_area(box1)
  area2 = cal_area(box2)

  upleft = np.maximum(box1[0], box2[0])
  downright = np.minimum(box1[1], box2[1])

  intersect = np.maximum((.0, .0), downright - upleft)

  intersect_area = intersect[0]*intersect[1]
  union_area = area1 + area2 - intersect_area

  return float(intersect_area)/union_area

def cvt_bbox2cwh(bbox):
  upleft, downright = bbox
  x1, y1 = upleft
  x2, y2 = downright

  x, y = (x1 + x2)/2.0, (y1 + y2)/2.0
  w, h = (x2 - x1), (y2 - y1)
  
  return ((x, y), (w, h))

def cvt_cwh2bbox(cwh):
  xy, wh = cwh 
  x, y = xy 
  w, h = wh

  x1, y1 = x - w/2.0, y - h/2.0
  x2, y2 = x + w/2.0, y + h/2.0

  return ((x1, y1), (x2, y2))

def img_listup(imgs):
  size = len(imgs)
  (h, w) = imgs[0].shape[:2]

  total_w = 0
  for img in imgs:
    total_w += img.shape[1]
  out = np.zeros((h, total_w, 3), np.uint8)

  offset = 0
  for i in range(size):
    h, w = imgs[i].shape[:2]
    out[:h, offset: offset + w] = imgs[i]
    offset += w

  return out

def visualization_orig(img, _annot, idx2obj, palette):
  print("visualization_orig()")
  h, w = img.shape[:2]

  _w, _h = _annot[0, :2]
  scale_w, scale_h = float(w)/_w, float(h)/_h
  for box in _annot[1:]:
    idx, x1, x2, y1, y2 = box
    idx = int(idx)
    _color = palette[idx]
    color = (int(_color[2]), int(_color[1]), int(_color[0]))

    name = idx2obj[idx]

    cx, cy = (scale_w*(x1 + x2)/2.0, scale_h*(y1 + y2)/2.0)

    b = (int(scale_w*x1), int(scale_h*y1))
    e = (int(scale_w*x2), int(scale_h*y2))
    img = cv2.rectangle(img, b, e, color, 5)
    img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    img = cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

  return img

