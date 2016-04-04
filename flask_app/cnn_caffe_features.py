import caffe
import numpy as np
from skimage import color, data, transform, filters
import pre_processing

# Launch caffe net with specific model
def set_caffe():
    
    model = 'bvlc_alexnet'
    width = 227
    height = 227

    caffe_root = '/home/ccampana/Documents/Install/Caffe/caffe/'
    path_2_model = caffe_root + 'models/' + model + '/'

    caffe.set_mode_cpu()
    net = caffe.Net(path_2_model + 'deploy.prototxt', path_2_model + \
    model + '.caffemodel', caffe.TEST)

    transformer = \
    caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_root + \
    'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    net.blobs['data'].reshape(1, 3, width, height)
    
    return net, transformer

# Computing feature vector out of caffe net
def get_features_caffe(image_name,net,transformer,color_flag):

    dim = 4096
    layer = 'fc6'
    size = 360 #227 #360
    #img = caffe.io.load_image(image_name)
    img = data.imread(image_name)
    img = transform.resize(img,(size,size),mode='nearest')
    if(color_flag == 'GRAY'):
      imgr = color.rgb2gray(img)
      imgr_inv = pre_processing.set_invariant_rep(imgr)
      img_cnn = \
      np.empty((imgr_inv.shape[0],imgr_inv.shape[1],3)).astype(imgr.dtype)
      img_cnn[:,:,0] = img_cnn[:,:,1] = img_cnn[:,:,2] = imgr_inv
    net.blobs['data'].data[...] = transformer.preprocess('data', img_cnn)
    out = net.forward()
    feature_vec = net.blobs[layer].data.reshape((dim,))
    feature_vec /= np.sum(feature_vec)

    return feature_vec


