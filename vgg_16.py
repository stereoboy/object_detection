
import tensorflow as tf

def conv_relu(tensor, W, B):
  conved = tf.nn.conv2d(tensor, W, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  biased = tf.nn.bias_add(conved, B, data_format='NCHW')
  relued = tf.nn.relu(biased)

  return relued


def init_VGG16(pretrained, channel=3):
  def load_weight(name):
    print(pretrained[name]['weights'].shape)
    return tf.constant_initializer(value=pretrained[name]['weights'])

  def load_bias(name):
    print(pretrained[name]['biases'].shape)
    return tf.constant_initializer(value=pretrained[name]['biases'])

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
