import numpy as np
import os
import random
import xmltodict
import cv2
from PIL import Image

idx2obj = [
  'blank',
	'aeroplane',
	'bicycle',
	'bird',
	'boat',
	'bottle',
	'bus',
	'car',
	'cat',
	'chair',
	'cow',
	'diningtable',
	'dog',
	'horse',
	'motorbike',
	'person',
	'pottedplant',
	'sheep',
	'sofa',
	'train',
	'tvmonitor',
]

def build_obj2idx(idx2obj):
  obj2idx = dict()
  for obj in idx2obj:
    obj2idx[obj] = len(obj2idx)
  
  for k, v in obj2idx.items():
    print(k, v)
  return obj2idx

def build_colormap_lookup(N):

  colormap = {}
  palette = np.zeros((N, 3), np.uint8)

  for i in range(0, N):
    ID = i 
    r = 0
    g = 0
    b = 0
    for j in range(0, 8):
      r = r | (((ID&0x1)>>0) <<(7-j))
      g = g | (((ID&0x2)>>1) <<(7-j))
      b = b | (((ID&0x4)>>2) <<(7-j))
      ID = ID >> 3

    colormap[(r,g,b)] = i
    palette[i, 0] = r
    palette[i, 1] = g
    palette[i, 2] = b

  palette = np.array(palette, np.uint8).reshape(-1, 3)
  global hidden_palette
  hidden_palette = palette
  return colormap, palette

def maybe_download(directory, filename, url):
  print('Try to dwnloaded', url)
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def load_pretrained(filepath):
  return np.load(filepath, encoding='bytes').item()

def img_listup(imgs):
  size = len(imgs)
  (h, w) = imgs[0].shape[:2]
  out = np.zeros((h, w*size, 3), np.uint8)

  offset = 0
  for i in range(size):
    h, w = imgs[i].shape[:2]
    out[:h, offset: offset + w] = imgs[i]
    offset += w
 
  return out

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

def cal_rel_coord((w, h), (x1, x2, y1, y2), (grid_x_size, grid_y_size)):
        
  cx, cy = ((x1 + x2)/2.0, (y1 + y2)/2.0)
  bw, bh = ((x2 - x1)/w, (y2 - y1)/h)

  x_loc = cx//grid_x_size
  cx = (cx - x_loc* grid_x_size)/grid_x_size - 0.5
  y_loc = cy//grid_y_size
  cy = (cy - y_loc* grid_y_size)/grid_y_size - 0.5

  return (int(x_loc),int( y_loc)), (cx, cy, bw, bh)


def visualization(img, _annot, num_grid, palette):
  print("visualization()")
  h, w = img.shape[:2]
  print(h, w)
  print(_annot)

  grid_size = float(h)/num_grid

  w_grid, h_grid = (w/num_grid, h/num_grid)
  _w, _h = _annot[0, :2]
  scale_w, scale_h = float(w)/_w, float(h)/_h 
  for box in _annot[1:]:
    idx, x1, x2, y1, y2 = box
    idx = int(idx)
    _color = palette[idx]
    color = (int(_color[2]), int(_color[1]), int(_color[0]))
    print("color:{}".format(color))
    
    name = idx2obj[idx]

    cx, cy = (scale_w*(x1 + x2)/2.0, scale_h*(y1 + y2)/2.0)
    x_loc = cx//w_grid
    y_loc = cy//h_grid

    grid_b = (int(grid_size*x_loc), int(grid_size*y_loc))
    grid_e = (int(grid_size*(x_loc+1)), int(grid_size*(y_loc+1)))

    b = (int(scale_w*x1), int(scale_h*y1))
    e = (int(scale_w*x2), int(scale_h*y2))
    #img = cv2.rectangle(img, grid_b, grid_e, color, -1)
    img = cv2.rectangle(img, b, e, color, 5)
    img = cv2.putText(img, name, b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    img = cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

  return img

def visualization2(img, annots):

  return img
