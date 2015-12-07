import numpy as np
# Make sure that caffe is on the python path:
caffe_root = '/home/joel/caffe/'  # this file is expected to be in {caffe_root}/examples
#caffe_root = '/home/ubuntu/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# GPU mode
caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGBi

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
# set net to batch size of 50
batch_size=6 #was 6!
net.blobs['data'].reshape(batch_size,3,224,224)

def set_batch_size(bs):
 global batch_size
 batch_size=bs
 net.blobs['data'].reshape(batch_size,3,224,224)

def run_image(imgs,debug=False,novelty=False):
 #net.blobs['data'].data[...] = transformer.preprocess('data', img)
 net.blobs['data'].data[...] = np.asarray([transformer.preprocess('data',x) for x in imgs])
 out = net.forward()

 #print len(net.blobs['pool5/7x7_s1'].data) #[0].shape
 #afds
 
 if debug:
  print("Predicted class is #{}.".format(out['prob'].argmax()))
  print out['prob'].max()
  # sort top k predictions from softmax output
  top_k = net.blobs['prob'].data[0].flatten()
  top_k_sort = top_k.argsort()[-1:-6:-1]
  print labels[top_k_sort],

 if novelty:
  return net.blobs['pool5/7x7_s1'].data

 return out['prob'],net.blobs['pool5/7x7_s1'].data

if (__name__=='__main__'):
 #out=run_image([caffe.io.load_image('/home/joel/mushroom.png')],debug=True)

 #jellyfish, confidence 0.14
 #hourglass, confidence 0.999923
 #piggybank, confidence 0.999991
 #out=run_image([caffe.io.load_image('/home/joel/Documents/jellyfish-thingiverse.png')],debug=True)

 #
 import time
 before=time.time()
 for k in range(5):
  out=run_image([caffe.io.load_image('/home/joel/Documents/hourglass-thingiverse.png')],debug=True)
  out=run_image([caffe.io.load_image('/home/joel/Documents/goblet-thingiverse.png')],debug=True)
  out=run_image([caffe.io.load_image('/home/joel/Documents/piggybank-thingiverse.jpg')],debug=True)
 after=time.time()
 print "---"
 print (after-before)/5.0
