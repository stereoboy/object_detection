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
import voc
import common
import multiprocessing
from tqdm import tqdm

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("B", "2", "number of Bound Box in grid cell")
tf.flags.DEFINE_integer("num_grid", "7", "number of grids vertically, horizontally")
tf.flags.DEFINE_integer("nclass", "20", "class num")
tf.flags.DEFINE_integer("img_size", "448", "sample image size")
# resize 538x538 (538=448*1.2)
tf.flags.DEFINE_integer("resize_size", "538", "size for augmentation")
tf.flags.DEFINE_integer("final_size", "646", "final image size")
tf.flags.DEFINE_float("resize_factor", "1.2", "sample image size")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img2", "directory for training data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot2", "directory for training data")
tf.flags.DEFINE_string("test_img_dir", "./test_img", "directory for test data")
tf.flags.DEFINE_string("test_annot_dir", "./test_annot", "directory for test data")

info_subpath  = "ImageSets/Main/"
annot_subpath = "Annotations/"
img_subpath   = "JPEGImages/"

def build_data_info_list(train_datasets, descfile='trainval.txt'):
  data_info_list = []
  for dataset in train_datasets:
    data_infos = {}
    data_infos['img_path'] = os.path.join(dataset, img_subpath)
    data_infos['annot_path'] = os.path.join(dataset, annot_subpath)
    info_path = os.path.join(dataset, info_subpath)
    with open(os.path.join(info_path, descfile)) as f:
      _filelist = f.readlines()
      filelist = [filename.rstrip() for filename in _filelist]
      data_infos['filelist'] = filelist
      data_info_list.append(data_infos)

  return data_info_list

def build_train_annot(xml_data, obj2idx):
  #print xml_data
  o = xmltodict.parse(xml_data)
  objs = o['annotation']['object']
  size = o['annotation']['size']
  w, h = (int(size['width']), int(size['height']))

  if isinstance(objs, list): # if a image have multiple objects
    annots = objs
  else: # only one object
    annots = [objs]

  annot_data = np.zeros((len(annots) + 1, 5), np.float32)

  # build header
  annot_data[0, 0] = FLAGS.resize_factor*w
  annot_data[0, 1] = FLAGS.resize_factor*h
  for i, annot in enumerate(annots):
    x1, y1 = (int(annot['bndbox']['xmin']), int(annot['bndbox']['ymin']))
    x2, y2 = (int(annot['bndbox']['xmax']), int(annot['bndbox']['ymax']))
    idx = obj2idx[annot['name']]

    offset_x = w*(FLAGS.resize_factor - 1.0)/2
    offset_y = h*(FLAGS.resize_factor - 1.0)/2
    annot_data[i+1, 0] = idx
    annot_data[i+1, 1] = x1 + offset_x
    annot_data[i+1, 2] = x2 + offset_x
    annot_data[i+1, 3] = y1 + offset_y
    annot_data[i+1, 4] = y2 + offset_y

  return annot_data

def build_test_annot(xml_data, obj2idx):
  #print xml_data
  o = xmltodict.parse(xml_data)
  objs = o['annotation']['object']
  size = o['annotation']['size']
  w, h = (int(size['width']), int(size['height']))

  if isinstance(objs, list): # if a image have multiple objects
    annots = objs
  else: # only one object
    annots = [objs]

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

  return annot_data

def preprocess_img(img):
  resized_img = cv2.resize(img, (FLAGS.resize_size, FLAGS.resize_size))
  margin = (FLAGS.final_size - FLAGS.resize_size)//2
  #final_img = cv2.copyMakeBorder(resized_img, margin, margin, margin, margin, cv2.BORDER_REFLECT_101)
  final_img = cv2.copyMakeBorder(resized_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
  return final_img

def build_train_data(data_info_list, obj2idx):

  p = multiprocessing.Pool(4)
  for data_infos in data_info_list:
    img_path    = data_infos['img_path']
    annot_path  = data_infos['annot_path']
    filelist    = data_infos['filelist']

    for filename in tqdm(filelist, desc="538x538 resize build annotation..."):

      # resize 538x538 (538=448*1.2)
      jpg_file = os.path.join(img_path, filename + '.jpg')
      img = cv2.imread(jpg_file)
      final_img = preprocess_img(img)
      cv2.imwrite(os.path.join(FLAGS.train_img_dir, filename + '.png'), final_img)

      # build annotation
      xml_file = os.path.join(annot_path, filename + '.xml')
      with open(xml_file) as f:
        xml_data = f.read()
        annot_data = build_train_annot(xml_data, obj2idx)
        np.save(os.path.join(FLAGS.train_annot_dir, filename), annot_data)

  return

def build_test_data(data_info_list, obj2idx):

  for data_infos in data_info_list:
    img_path    = data_info['img_path']
    annot_path  = data_info['annot_path']
    filelist    = data_info['filelist']

    for filename in tqdm(filelist, desc="538x538 resize build annotation..."):

      # resize 538x538 (538=448*1.2)
      jpg_file = os.path.join(img_path, filename + '.jpg')
      img = cv2.imread(jpg_file)
      #final_img = preprocess_img(img)
      cv2.imwrite(os.path.join(FLAGS.train_img_dir, filename + '.png'), final_img)

      # build annotation
      xml_file = os.path.join(annot_path, filename + '.xml')
      with open(xml_file) as f:
        xml_data = f.read()
        annot_data = build_test_annot(xml_data, obj2idx)
        np.save(os.path.join(FLAGS.train_annot_dir, filename), annot_data)

  return

def main(args):

  if not os.path.exists(FLAGS.train_img_dir):
    os.mkdir(FLAGS.train_img_dir)
  if not os.path.exists(FLAGS.train_annot_dir):
    os.mkdir(FLAGS.train_annot_dir)
  if not os.path.exists(FLAGS.test_img_dir):
    os.mkdir(FLAGS.test_img_dir)
  if not os.path.exists(FLAGS.test_annot_dir):
    os.mkdir(FLAGS.test_annot_dir)

  colormap, palette = voc.build_colormap_lookup(21)
  obj2idx = voc.build_obj2idx(voc.idx2obj)

  train_datasets = ['../../data/VOCdevkit/VOC2007', '../../data/VOCdevkit/VOC2012']
  test_datasets = ['../../data/VOCdevkit/VOC2007']

  train_info_list = build_data_info_list(train_datasets, 'trainval.txt')
  test_info_list  = build_data_info_list(test_datasets, 'test.txt')

  #print(train_info_list)
  #print(test_info_list)

  build_train_data(train_info_list, obj2idx)
  build_test_data(test_info_list, obj2idx)

  if not os.path.isfile(FLAGS.filelist):
    with open(FLAGS.filelist, "w+") as out:

      train_list = [os.path.basename(path) for path in train_img_list]
      test_list  = [os.path.basename(path) for path in test_img_list]

      data = {'train':train_list, 'test':test_list}
      json.dump(data, out, indent=2)

if __name__ == "__main__":
  tf.app.run()
