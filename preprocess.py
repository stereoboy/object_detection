import tensorflow as tf
import numpy as np
import glob
import os
import xmltodict
import json
from datetime import datetime, date, time
import cv2
import sys
import getopt
import pickle
import common

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("B", "2", "number of Bound Box in grid cell")
tf.flags.DEFINE_integer("num_grid", "7", "number of grids vertically, horizontally")
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_integer("img_size", "448", "sample image size")
tf.flags.DEFINE_integer("grid_size", "64", "grid size")
tf.flags.DEFINE_integer("num_threads", "6", "max thread number")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")


def cal_rel_coord((x, y), (grid_x_size, grid_y_size)):
  
  x_loc = x//grid_x_size
  x = float(x - x_loc* grid_x_size)/grid_x_size - 0.5
  y_loc = y//grid_y_size
  y = float(y - y_loc* grid_y_size)/grid_y_size - 0.5

  return (x, y), (int(x_loc),int( y_loc))

def main(args):

  if not os.path.exists(FLAGS.train_img_dir):
    os.mkdir(FLAGS.train_img_dir)
  if not os.path.exists(FLAGS.train_annot_dir):
    os.mkdir(FLAGS.train_annot_dir)


  obj2idx = common.build_obj2idx(common.idx2obj)
  colormap, palette = common.build_colormap_lookup(21)

  info_path = FLAGS.data_dir + "ImageSets/Main/"
  annot_path = FLAGS.data_dir + "Annotations/"
  img_path = FLAGS.data_dir + "JPEGImages/"

  
  with open(os.path.join(info_path, 'trainval.txt')) as f:
    _filelist = f.readlines()
    filelist = [ filename.rstrip() for filename in _filelist] 

  #if not os.path.isfile(FLAGS.filelist):
  with open(FLAGS.filelist, "wb+") as out:
    data = filelist
    json.dump(data, out, indent=2)

  annot_list = [] 
  for filename in filelist:
    print filename
    xml_file = os.path.join(annot_path, filename + '.xml')
    with open(xml_file) as f:
      xml_data = f.read()
      #print xml_data
      o = xmltodict.parse(xml_data)
      objs = o['annotation']['object']
      size = o['annotation']['size']
      w, h = (int(size['width']), int(size['height']))
      w_grid = float(w)/FLAGS.num_grid
      h_grid = float(h)/FLAGS.num_grid


      if isinstance(objs, list): # if a image have multiple objects
        annots = objs
      else: # only one object
        annots = [objs]
      annot_list.append(annots)

      jpg_file = os.path.join(img_path, filename + '.jpg')
      img = cv2.imread(jpg_file)
      vis = img.copy()

      vis_grid = np.zeros((FLAGS.img_size, FLAGS.img_size, 3), np.uint8)
      resized_img = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))

      cv2.imwrite(os.path.join(FLAGS.train_img_dir, filename + '.png'), resized_img)

      # draw grid
      for i in range(FLAGS.num_grid):
        for j in range(FLAGS.num_grid):
          b = (int(i*w_grid), int(j*h_grid))
          e = (int((i + 1)*w_grid), int((j + 1)*h_grid))
          vis = cv2.rectangle(vis, b, e, (0, 0, 0), 2)

      _annot_data = {}
      for annot in annots:
        print annot
        b = (int(annot['bndbox']['xmin']), int(annot['bndbox']['ymin']))
        e = (int(annot['bndbox']['xmax']), int(annot['bndbox']['ymax']))
        idx = obj2idx[annot['name']]
        _color = palette[idx]
        color = (int(_color[2]), int(_color[1]), int(_color[0]))
        print b, e, color
        print color

        x1, y1 = b
        x2, y2 = e
        cx, cy = ((x1 + x2)/2.0, (y1 + y2)/2.0)

        bw, bh = (float(x2 - x1)/w, float(y2 - y1)/h)

        (x, y), (x_loc, y_loc) = cal_rel_coord((cx, cy), (w_grid, h_grid))

        if not (x_loc, y_loc) in _annot_data.keys():
          _annot_data[(x_loc, y_loc)] = (idx, [])

        if _annot_data[(x_loc, y_loc)][0] == idx:
          _annot_data[(x_loc, y_loc)][1].append((1.0, x, y, bw, bh))

          print _annot_data

          grid_b = (int(FLAGS.grid_size*x_loc), int(FLAGS.grid_size*y_loc))
          grid_e = (int(FLAGS.grid_size*(x_loc+1)), int(FLAGS.grid_size*(y_loc+1)))

          vis_grid = cv2.rectangle(vis_grid, grid_b, grid_e, color, -1)
          vis = cv2.rectangle(vis, b, e, color, 5)
          vis = cv2.putText(vis, annot['name'], b, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
          vis = cv2.circle(vis, (int(cx), int(cy)), 4, color, -1)

          resized_img = cv2.circle(resized_img, (int(FLAGS.grid_size*(x_loc + x)), int(FLAGS.grid_size*(y_loc + y))), 4, color, -1)

      cell_info_dim = FLAGS.nclass + FLAGS.B*(1 + 4) # 2x(confidence + (x, y, w, h)) + class

      annot_data = np.zeros((FLAGS.num_grid, FLAGS.num_grid, cell_info_dim), np.float32)
      for (x_loc, y_loc), (idx, bbs)  in _annot_data.items():
        print (x_loc, y_loc), (idx, bbs)
        annot_data[y_loc, x_loc, idx-1] = 1
        for i in range(min(2, len(bbs))):
          b = FLAGS.nclass + (1 + 4)*i
          e = b + (1 + 4)
          annot_data[y_loc, x_loc, b:e] = np.array(bbs[i], np.float32)

      with open(os.path.join(FLAGS.train_annot_dir, filename + '.label'), 'wb') as f:
        f.write(annot_data.tobytes())

#      print "========================================================================="
#      with open(os.path.join(FLAGS.train_annot_dir, filename + '.label'), 'rb') as f:
#        annot_data = np.frombuffer(f.read(), dtype=np.float32)
#        annot_data = annot_data.reshape((FLAGS.num_grid, FLAGS.num_grid, -1))
#
#      for i in range(FLAGS.num_grid):
#        for j in range(FLAGS.num_grid):
#          print (i, j), annot_data[i, j]

      resized_img = 0.7*resized_img + 0.3*vis_grid
      resized_img = resized_img.astype(np.uint8)
#      cv2.imshow('jpeg', vis)
#      cv2.imshow('resized', resized_img)
#
#      cv2.waitKey(0)

if __name__ == "__main__":
  tf.app.run()
