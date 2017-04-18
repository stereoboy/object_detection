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
from tqdm import tqdm

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("B", "2", "number of Bound Box in grid cell")
tf.flags.DEFINE_integer("num_grid", "7", "number of grids vertically, horizontally")
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_integer("img_size", "448", "sample image size")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "base directory for data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "base directory for data")

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
  with open(FLAGS.filelist, "w+") as out:
    data = [filename for filename in filelist]
    json.dump(data, out, indent=2)

  max_object = -1
  annot_list = []
  for filename in tqdm(filelist, desc="537x537 resize build annotation..."):

    # resize 537x537 (537=448*1.2)
    jpg_file = os.path.join(img_path, filename + '.jpg')
    img = cv2.imread(jpg_file)
    resized_img = cv2.resize(img, (537, 537))
    cv2.imwrite(os.path.join(FLAGS.train_img_dir, filename + '.png'), resized_img)

    # build annotation
    xml_file = os.path.join(annot_path, filename + '.xml')
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
      annot_list.append(annots)

      if max_object < len(annots):
        max_object = len(annots)

      annot_data = np.zeros((len(annots) + 1, 5), np.float32)
      
      # build header
      annot_data[0, 0] = w
      annot_data[0, 1] = h
      for i, annot in enumerate(annots):
        x1, y1 = (int(annot['bndbox']['xmin']), int(annot['bndbox']['ymin']))
        x2, y2 = (int(annot['bndbox']['xmax']), int(annot['bndbox']['ymax']))
        idx = obj2idx[annot['name']]

        annot_data[i+1, 0] = idx
        annot_data[i+1, 1] = x1
        annot_data[i+1, 2] = x2
        annot_data[i+1, 3] = y1
        annot_data[i+1, 4] = y2 

      np.save(os.path.join(FLAGS.train_annot_dir, filename), annot_data)

if __name__ == "__main__":
  tf.app.run()
