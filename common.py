import numpy as np
import os
import random
import xmltodict
import cv2
from PIL import Image

class DataCenter(object):
  def nextPair(self):
    raise NotImplementedError

  def shuffle(self):
    raise NotImplementedError

  def getPair(self, idx):
    raise NotImplementedError

  @property
  def size(self):
    raise NotImplementedError

class VOC2012(DataCenter):
  def __init__(self, train_val_ratio=0.1):

    base_path = "../../data/VOCdevkit/VOC2012/"
    info_path = base_path + "ImageSets/Main/"
    self.img_path = base_path + "JPEGImages/"
    self.annot_path = base_path + "Annotations/"

    with open(os.path.join(info_path, "trainval.txt")) as f:
      _filelist = f.readlines()
      filelist = [ filename.rstrip() for filename in _filelist]

    self._size = int(train_val_ratio*len(filelist))
    self._val_size = len(filelist) - self._size

    trainpairs = filelist[:self._size]
    validpairs = filelist[self._size:]

    self.trainpairs = trainpairs
    self.validpairs = validpairs

  def shuffle(self):
    random.shuffle(self.trainpairs)

  @property
  def size(self):
    return self._size

  @property
  def val_size(self):
    return self._val_size

  def load_annot(self, path):

    xml_file = os.path.join(path)
    with open(xml_file) as f:
      xml_data = f.read()
      #print xml_data
      o = xmltodict.parse(xml_data)
      objs = o['annotation']['object']
      size = o['annotation']['size']
      w, h = (int(size['width']), int(size['height']))

      if isinstance(objs, list): # if a image have multiple objects
        annots = objs
      else: # only one object
        annots = [objs]

    return ((w, h), annots)

  def _getPair(self, pairs, idx):
    filename = pairs[idx]

    jpegpath = os.path.join(self.img_path, filename + '.jpg')
    annotpath = os.path.join(self.annot_path, filename + '.xml')
    _img = Image.open(jpegpath)
    label = self.load_annot(annotpath)

    img = np.array(_img)
    _img.close()
    return (filename, img, label)

  def getTrainPair(self, idx):
    return self._getPair(self.trainpairs, idx)

  def getValPair(self, idx):
    return self._getPair(self.validpairs, idx)

def cal_rel_coord(w, h, x1, x2, y1, y2, w_grid, h_grid):
  #print('cal_rel_coord')
  #print(w,h, x1, x2, y1, y2, w_grid, h_grid)
  cx, cy = ((x1 + x2)/2.0, (y1 + y2)/2.0)
  nw, nh = ((x2 - x1)/w, (y2 - y1)/h)

  x_loc = cx//w_grid
  cx = (cx - x_loc*w_grid)/w_grid
  y_loc = cy//h_grid
  cy = (cy - y_loc*h_grid)/h_grid

  return (int(x_loc),int( y_loc)), (cx, cy, nw, nh)

def logit(x):
  return np.log(x) - np.log(1 - x)

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  #e_x = np.exp(x - np.max(x))
  e_x = np.exp(x)
  return e_x / e_x.sum()

def clip(x, min_val, max_val):
  return max(min_val, min(max_val, x))
