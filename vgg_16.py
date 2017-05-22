import os
import sys

import tensorflow as tf
import tarfile

from six.moves import urllib

# this code for pre-trained vgg front end
# based on slim/nets/vgg.py

slim = tf.contrib.slim

def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
checkpoints_dir = './vgg_16_ckpts'

def setup_vgg_16():

  if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)
  if not tf.gfile.Exists(os.path.join(checkpoints_dir, 'vgg_16.ckpt')):
		download_and_uncompress_tarball(url, checkpoints_dir)

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    #with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME', data_format='NCHW') as arg_sc:
    #with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16_base(inputs,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points

def conv_relu(tensor, W, B):
  conved = tf.nn.conv2d(tensor, W, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  biased = tf.nn.bias_add(conved, B, data_format='NCHW')
  relued = tf.nn.relu(biased)

  return relued


def init_VGG16(pretrained, channel=3):
  def load_weight(name):
    print(pretrained[name][b'weights'].shape)
    return tf.constant_initializer(value=pretrained[name][b'weights'])

  def load_bias(name):
    print(pretrained[name][b'biases'].shape)
    return tf.constant_initializer(value=pretrained[name][b'biases'])

  # initialize with Pretrained model
  Ws = {
      "1_1":tf.get_variable('conv1_1', shape = [3, 3, channel, 64], initializer=load_weight('conv1_1')),
      "1_2":tf.get_variable('conv1_2', shape = [3, 3, 64, 64], initializer=load_weight('conv1_2')),

      "2_1":tf.get_variable('conv2_1', shape = [3, 3, 64, 128], initializer=load_weight('conv2_1')),
      "2_2":tf.get_variable('conv2_2', shape = [3, 3, 128, 128], initializer=load_weight('conv2_2')),

      "3_1":tf.get_variable('conv3_1', shape = [3, 3, 128, 256], initializer=load_weight('conv3_1')),
      "3_2":tf.get_variable('conv3_2', shape = [3, 3, 256, 256], initializer=load_weight('conv3_2')),
      "3_3":tf.get_variable('conv3_3', shape = [3, 3, 256, 256], initializer=load_weight('conv3_3')),

      "4_1":tf.get_variable('conv4_1', shape = [3, 3, 256, 512], initializer=load_weight('conv4_1')),
      "4_2":tf.get_variable('conv4_2', shape = [3, 3, 512, 512], initializer=load_weight('conv4_2')),
      "4_3":tf.get_variable('conv4_3', shape = [3, 3, 512, 512], initializer=load_weight('conv4_3')),

      "5_1":tf.get_variable('conv5_1', shape = [3, 3, 512, 512], initializer=load_weight('conv5_1')),
      "5_2":tf.get_variable('conv5_2', shape = [3, 3, 512, 512], initializer=load_weight('conv5_2')),
      "5_3":tf.get_variable('conv5_3', shape = [3, 3, 512, 512], initializer=load_weight('conv5_3')),
      }

  Bs = {
      "1_1":tf.get_variable('bias1_1', shape = [64], initializer=load_bias('conv1_1')),
      "1_2":tf.get_variable('bias1_2', shape = [64], initializer=load_bias('conv1_2')),

      "2_1":tf.get_variable('bias2_1', shape = [128], initializer=load_bias('conv2_1')),
      "2_2":tf.get_variable('bias2_2', shape = [128], initializer=load_bias('conv2_2')),

      "3_1":tf.get_variable('bias3_1', shape = [256], initializer=load_bias('conv3_1')),
      "3_2":tf.get_variable('bias3_2', shape = [256], initializer=load_bias('conv3_2')),
      "3_3":tf.get_variable('bias3_3', shape = [256], initializer=load_bias('conv3_3')),

      "4_1":tf.get_variable('bias4_1', shape = [512], initializer=load_bias('conv4_1')),
      "4_2":tf.get_variable('bias4_2', shape = [512], initializer=load_bias('conv4_2')),
      "4_3":tf.get_variable('bias4_3', shape = [512], initializer=load_bias('conv4_3')),

      "5_1":tf.get_variable('bias5_1', shape = [512], initializer=load_bias('conv5_1')),
      "5_2":tf.get_variable('bias5_2', shape = [512], initializer=load_bias('conv5_2')),
      "5_3":tf.get_variable('bias5_3', shape = [512], initializer=load_bias('conv5_3')),
      }
  return Ws, Bs

def model_VGG16(x, Ws, Bs):

  mp_ksize= [1, 1, 2, 2]
  mp_strides=[1, 1, 2, 2]

  relued = conv_relu(x, Ws['1_1'], Bs['1_1'])
  relued = conv_relu(relued, Ws['1_2'], Bs['1_2'])
  pooled1 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled1, Ws['2_1'], Bs['2_1'])
  relued = conv_relu(relued, Ws['2_2'], Bs['2_2'])
  pooled2 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled2, Ws['3_1'], Bs['3_1'])
  relued = conv_relu(relued, Ws['3_2'], Bs['3_2'])
  relued = conv_relu(relued, Ws['3_3'], Bs['3_3'])
  pooled3 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled3, Ws['4_1'], Bs['4_1'])
  relued = conv_relu(relued, Ws['4_2'], Bs['4_2'])
  relued = conv_relu(relued, Ws['4_3'], Bs['4_3'])
  pooled4 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  relued = conv_relu(pooled4, Ws['5_1'], Bs['5_1'])
  relued = conv_relu(relued, Ws['5_2'], Bs['5_2'])
  relued = conv_relu(relued, Ws['5_3'], Bs['5_3'])
  pooled5 = tf.nn.max_pool(relued, ksize=mp_ksize, strides=mp_strides, padding='SAME', data_format='NCHW')

  return pooled5
