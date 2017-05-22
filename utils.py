import tensorflow as tf
import numpy as np
from PIL import Image
import os

def get_epoch():
  epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
  epoch_update = epoch_step.assign(epoch_step + 1)
  return epoch_step, epoch_update

def load_imgs(train_img_dir, filelist):
  def load_img(path):
    _img = Image.open(path)
    img = np.array(_img)
    _img.close()
    return img

  _imgs = [os.path.join(train_img_dir, filename + ".png") for filename in filelist]

  imgs = [load_img(_img) for _img in _imgs]
  return imgs

def load_annots(train_annot_dir, filelist):
  def load_annot(path):
    #print(path)
    annot = np.load(path, encoding='bytes')
    #print("original dims: {}x{}".format(annot[0,0], annot[0,1]))
    return  annot

  _annots = [os.path.join(train_annot_dir, filename + ".npy") for filename in filelist]

  annots = [load_annot(_annot) for _annot in _annots]

  return annots

