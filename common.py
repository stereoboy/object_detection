import numpy as np

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
    print k, v
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
  print 'Try to dwnloaded', url
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print 'Successfully downloaded', filename, size, 'bytes.'
  return filepath

def load_pretrained(filepath):
  return np.load(filepath).item()
