import tensorflow as tf
import numpy as np
import glob
import os
import xmltodict
import json
from datetime import datetime, date, time
import cv2
import sys
import signal
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
tf.flags.DEFINE_integer("mp_num", "5", "numbers of multiprocessing")
tf.flags.DEFINE_float("resize_factor", "1.2", "sample image size")
tf.flags.DEFINE_string("filelist", "filelist.json", "filelist.json")
tf.flags.DEFINE_string("balanced_filelist", "balanced.json", "normalized filelist")
tf.flags.DEFINE_string("data_dir", "../../data/VOCdevkit/VOC2012/", "base directory for data")
tf.flags.DEFINE_string("train_img_dir", "./train_img", "directory for training data")
tf.flags.DEFINE_string("train_annot_dir", "./train_annot", "directory for training data")
tf.flags.DEFINE_string("test_img_dir", "./test_img", "directory for test data")
tf.flags.DEFINE_string("test_annot_dir", "./test_annot", "directory for test data")

info_subpath  = "ImageSets/Main/"
annot_subpath = "Annotations/"
img_subpath   = "JPEGImages/"

def build_data_info_list(datasets, descfile='trainval.txt'):
  data_info_list = []
  total_list = []
  for dataset in datasets:
    data_infos = {}
    data_infos['img_path'] = os.path.join(dataset, img_subpath)
    data_infos['annot_path'] = os.path.join(dataset, annot_subpath)
    info_path = os.path.join(dataset, info_subpath)
    with open(os.path.join(info_path, descfile)) as f:
      _filelist = f.readlines()
      filelist = [filename.rstrip() for filename in _filelist]
      data_infos['filelist'] = filelist
      data_info_list.append(data_infos)
      total_list.extend(filelist)
  return data_info_list, total_list

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

def init_worker():
  import signal
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def work_train_func(args):
  i, filelist, img_path, annot_path, obj2idx= args
  print("mp[{}] start".format(i))

  for j, filename in enumerate(filelist):
    if j > 0 and j%100 == 0:
      print('mp[{}] has done {} files.'.format(i, j))

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
  print("mp[{}] done".format(i))

def work_test_func(args):
  i, filelist, img_path, annot_path, obj2idx= args
  print("mp[{}] start".format(i))

  for j, filename in enumerate(filelist):
    if j > 0 and j%100 == 0:
      print('mp[{}] has done {} files.'.format(i, j))

    jpg_file = os.path.join(img_path, filename + '.jpg')
    img = cv2.imread(jpg_file)
    cv2.imwrite(os.path.join(FLAGS.test_img_dir, filename + '.png'), img)

    # build annotation
    xml_file = os.path.join(annot_path, filename + '.xml')
    with open(xml_file) as f:
      xml_data = f.read()
      annot_data = build_test_annot(xml_data, obj2idx)
      np.save(os.path.join(FLAGS.test_annot_dir, filename), annot_data)
  print("mp[{}] done".format(i))

def build_data(data_info_list, obj2idx, work_func):

  for i, data_infos in tqdm(enumerate(data_info_list), desc='build data'):
    img_path    = data_infos['img_path']
    annot_path  = data_infos['annot_path']
    filelist    = data_infos['filelist']

    print("[{}] processing on dataset:{}".format(i, img_path))

    # build data for mp
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(FLAGS.mp_num)
    signal.signal(signal.SIGINT, original_sigint_handler)

    unit_size = int(np.ceil(float(len(filelist))/FLAGS.mp_num))
    mpdata = [(i, filelist[i*unit_size:(i + 1)*unit_size], img_path, annot_path, obj2idx)for i in range(FLAGS.mp_num)]

    start = datetime.now()
    try:
      pool.map_async(work_func, mpdata)
    except KeyboardInterrupt:
      print('keyboardInterrupt')
      pool.terminate()
      pool.join()

    pool.close()
    pool.join()

    current = datetime.now()
    print('\telapsed:' + str(current - start))

  return


def build_train_data(data_info_list, obj2idx):
  return build_data(data_info_list, obj2idx, work_train_func)


def build_test_data(data_info_list, obj2idx):
  return build_data(data_info_list, obj2idx, work_test_func)


def work_gather_histogram(args):
  i, filelist, obj2idx = args
  print("mp[{}] start".format(i))
  histogram = np.zeros(len(obj2idx), dtype=np.int32)

  for filename in filelist:
    path = os.path.join(FLAGS.train_annot_dir, filename + '.npy')
    annot = np.load(path, encoding='bytes')
    for i in range(1, len(annot)):
      idx = int(annot[i, 0])
      histogram[idx] += 1
  return histogram

def gather_histogram(filelist, obj2idx):
  histogram = np.zeros(len(obj2idx), dtype=np.int32)

  # build data for mp
  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  pool = multiprocessing.Pool(FLAGS.mp_num)
  signal.signal(signal.SIGINT, original_sigint_handler)

  unit_size = int(np.ceil(float(len(filelist))/FLAGS.mp_num))
  mpdata = [(i, filelist[i*unit_size:(i + 1)*unit_size], obj2idx) for i in range(FLAGS.mp_num)]

  start = datetime.now()
  try:
    histograms = pool.map(work_gather_histogram, mpdata)
  except KeyboardInterrupt:
    print('keyboardInterrupt')
    pool.terminate()
    pool.join()

  pool.close()
  pool.join()

  current = datetime.now()
  print('\telapsed:' + str(current - start))

  histogram = sum(list(histograms))
  return histogram

def analyze_label(datasets, obj2idx, descfile='trainval.txt'):

  _data_info_list = [[] for _ in range(len(obj2idx))]
  for dataset in datasets:
    for obj_name in obj2idx.keys():
      if obj_name == 'blank':
        continue

      idx = obj2idx[obj_name]
      data_infos = {}
      data_infos['label'] = obj_name
      data_infos['img_path'] = os.path.join(dataset, img_subpath)
      data_infos['annot_path'] = os.path.join(dataset, annot_subpath)
      info_path = os.path.join(dataset, info_subpath)
      filename = obj_name + '_' + descfile
      full_path = os.path.join(info_path, filename)
      with open(full_path) as f:
        print(full_path)
        _filelist = f.readlines()
        #filelist = [filename.split()[0] for filename in _filelist if filename.split()[1] != '-1']
        filelist = [filename.split()[0] for filename in _filelist if filename.split()[1] == '1']
        data_infos['filelist'] = filelist
        data_infos['count'] = len(filelist)
      _data_info_list[idx].append(data_infos)

  # merge
  label_info_list = [{} for _ in range(len(obj2idx))]
  for i in range(1, len(_data_info_list)):
    label_info_list[i]['label'] = _data_info_list[i][0]['label']
    filelist = []
    for data_infos in _data_info_list[i]:
      filelist.extend(data_infos['filelist'])
    label_info_list[i]['filelist'] = filelist
    label_info_list[i]['count'] = len(filelist)

    print(label_info_list[i]['label'], label_info_list[i]['count'])

  # analyze histogram in terms of person label
  print('------------------------------------------')
  person_id = obj2idx['person']
  personset = set(label_info_list[person_id]['filelist'])
  for i in range(1, len(label_info_list)):
    if i == obj2idx['person']:
      continue

    fileset = set(label_info_list[i]['filelist'])

    exclusive_set = fileset.difference(personset)
    label_info_list[i]['exclusive_set'] = exclusive_set
    label_info_list[i]['exclusive_set_count'] = len(exclusive_set)

    print(label_info_list[i]['label'], label_info_list[i]['exclusive_set_count'])

  return label_info_list

# hand-craft data after trial-error
duplicate_ratios = {
    'blank':0,
    'aeroplane':12,
    'bicycle': 34,
    'bird':8,
    'boat':13,
    'bottle':12,
    'bus':25,
    'car':2,
    'cat':6,
    'chair':0,
    'cow':16,
    'diningtable':20,
    'dog':6,
    'horse':27,
    'motorbike':30,
    'person':0,
    'pottedplant':7,
    'sheep':13,
    'sofa':11,
    'train':20,
    'tvmonitor':13,
}
def create_balanced_dataset(label_info_list, obj2idx):
  filelist = []
  person_id = obj2idx['person']
  person_count = label_info_list[person_id]['count']
  for i in range(1, len(label_info_list)):
    if i == obj2idx['person']:
      continue

    label = label_info_list[i]['label']
    count = label_info_list[i]['count']
    exclusive_count = label_info_list[i]['exclusive_set_count']
    diff = person_count - count
    duplicate_ratio = diff//exclusive_count
    print(label, "({} - {})//{}".format(person_count, count, exclusive_count), duplicate_ratio)

    exclusive_set = label_info_list[i]['exclusive_set']
    filelist.extend(duplicate_ratios[label]*list(exclusive_set))

  return filelist


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

  train_info_list, train_list = build_data_info_list(train_datasets, 'trainval.txt')
  test_info_list, test_list  = build_data_info_list(test_datasets, 'test.txt')

#  print(train_info_list)
#  print(test_info_list)
#
  build_train_data(train_info_list, obj2idx)
  build_test_data(test_info_list, obj2idx)

  with open(FLAGS.filelist, "w+") as out:
    data = {'train':train_list, 'test':test_list}
    json.dump(data, out, indent=2)

  np.set_printoptions(linewidth=200)
  label_info_list = analyze_label(train_datasets, obj2idx)
  additional_balanced_filelist = create_balanced_dataset(label_info_list, obj2idx)

  with open(FLAGS.balanced_filelist, "w+") as out:
    data = {'train':train_list + additional_balanced_filelist, 'test':test_list}
    json.dump(data, out, indent=2)


  print("## check results ##############")
  with open(FLAGS.filelist, "r") as f:
    train_list = json.load(f)['train']

  with open(FLAGS.balanced_filelist, "r") as f:
    balanced_train_list = json.load(f)['train']

  print('train_set: {}'.format(len(train_list)))
  histogram = np.zeros(len(obj2idx), dtype=np.int32)
  for filename in train_list:
    path = os.path.join(FLAGS.train_annot_dir, filename + '.npy')
    annot = np.load(path, encoding='bytes')
    for i in range(1, len(annot)):
      idx = int(annot[i, 0])
      histogram[idx] += 1

  for i in range(len(histogram)):
    print("{}:{}".format(voc.idx2obj[i], histogram[i]))

  print('balanced_train_set: {}'.format(len(balanced_train_list)))
  histogram = gather_histogram(balanced_train_list, obj2idx)
  for i in range(len(histogram)):
    print("{}:{}".format(voc.idx2obj[i], histogram[i]))
if __name__ == "__main__":
  tf.app.run()
