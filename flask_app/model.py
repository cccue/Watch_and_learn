from PIL import Image
import db 
import numpy as np
import os
import predict
from scipy import ndimage as ndi
from sklearn import cluster
from skimage import io, color, transform
from sklearn.externals import joblib
import cnn_caffe_features

net = None
transformer = None

# To initialize CNN only once
def get_cnn_objects():
    global net, transformer
    if net is None:
       net, transformer = cnn_caffe_features.set_caffe()

    return net, transformer

# Color quantization signature 
def call_color_sig(path_to_image):

    size = 360
    nclusters = 15

    image_array = io.imread(path_to_image)
    image_array = transform.resize(image_array,(size,size), \
    mode='nearest')
    image_array = color.rgb2lab(image_array)
    image_array = image_array.reshape(-1,3)

    k_means = cluster.MiniBatchKMeans(nclusters,init='k-means++',n_init=1,\
              max_iter=300,tol=0.01,random_state=451792)
    k_means.fit(image_array)
    centers = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_

    pixels_tot = 0
    pixels_loc = np.empty((nclusters,1),dtype=int)
    for index in np.arange(0,nclusters):
        pixels_loc[index] = np.sum((labels == index).astype('int'))
        pixels_tot += pixels_loc[index]

    weights = pixels_loc.astype('float')/pixels_tot

    #print "Total number of pixels ", pixels_tot
    signature = \
    np.concatenate((weights,centers),axis=1).T.flatten()

    return signature

# Remove background histogram
def remove_bckg(feature_vec,feature_matrix):

    feature_vec_red = np.delete(feature_vec,(126,127)).reshape(1, -1)
    feature_vec_red = \
    feature_vec_red.astype('float')/np.sum(feature_vec_red)

    feature_matrix_red = np.delete(feature_matrix,(126,127),axis=1)
    feature_matrix_red = feature_matrix_red.astype('float')/ \
    np.sum(feature_matrix_red,axis=1).reshape(-1, 1)

    return feature_vec_red, feature_matrix_red

# Color reduction algorithm
def reduce_colors(image,r_flag):

    if(r_flag == 0):
      num_colors = 27
    elif(r_flag == 1):
      # This is a 64 color scheme very similar to the one in pyccv
      num_colors = 64
    elif(r_flag == 1):
      # This is a 216 color Web safe palette 
      num_colors = 216

    num_bins = np.floor(np.power(np.log2(num_colors)/3,2)).astype('int')
    num_bins_sq = np.power(num_bins,2).astype('int')
    scale = 256/num_bins
    image_reduced = np.array(np.floor(image[:,:,0]/scale)*num_bins_sq + \
    np.floor(image[:,:,1]/scale)*num_bins + np.floor(image[:,:,2]/scale))

    max_range = np.floor(255/scale)*num_bins_sq + \
    np.floor(255/scale)*num_bins + np.floor(255/scale) + 0.5

    #image_reduced = np.array(image[:,:,0]/64*14 + image[:,:,1]/64*4 + \
    #image[:,:,2]/64)

    return image_reduced, max_range

# Computing color-coherence-vectors (CCV) for reference image
# This function follows my own implementation that reproduces
# the pyccv results at: https://pypi.python.org/pypi/pyccv 
# while avoiding its intallation and the python/C
# library interfacing
def call_pyccv_mine(path_to_image):

    # Normalization (This is the size of most
    # images obatined from Jomashop
    size = 360
    # Size thresholding as needed by the CCV algorithm
    threshold = 3000
    # Number of bins in CCV
    nbins = 64
   
    # Reading and resizing image 
    image_input = Image.open(path_to_image)
    image_input = image_input.convert(mode='RGB')
    image_input = image_input.resize((size, size), Image.ANTIALIAS)
    image_array = np.asarray(image_input)
    # Gaussian filtering
    sigma = 1.8
    image_filtered = np.empty(image_array.shape)
    for i in range(3):
        image_filtered[:, :, i] = \
        ndi.filters.gaussian_filter(image_array[:, :, i],sigma)
    img_out = image_filtered.astype('int')
    # Color palette reduction
    (image_reduced, max_range) = reduce_colors(img_out,1)
    (hist_pillow, indices) = \
    np.histogram(image_reduced,nbins,range=(-0.5,max_range))

    ccv_c = np.empty(nbins,dtype=int)
    ccv_i = np.empty(nbins,dtype=int)
    ccv = np.empty(2*nbins,dtype=int)
    for index in range(nbins):
        # Binning the color range
        local_matrix_bin = \
        (image_reduced >= indices[index]).astype('int') * \
        (image_reduced < indices[index+1]).astype('int')
        # Labeling connected components    
        connected_edges, connected_labels = ndi.label(local_matrix_bin)
        # Filtering background out (0-value pixels)
        d_labels = connected_edges.ravel()
        d_labels_non_zero = d_labels[d_labels != 0]
        # Obtaining the size distribution of connected components
        connected_sizes = \
        np.sort(np.bincount(d_labels_non_zero))
        # Splitting size distribution into coherent/incoherent pixels
        ccv_c[index] = np.sum(connected_sizes[connected_sizes > threshold])
        ccv_i[index] = np.sum(connected_sizes[connected_sizes <= threshold])

    # Assembling coherent and incoherent vectors into one entity 
    ccv[0::2] = ccv_c
    ccv[1::2] = ccv_i
    print " Number of pixels in input image: ", np.sum(ccv)

    return ccv

# Outputing list of first K-similar images as defined
# by the CCV feature space
def predict_similar_images(image_input):

    root_path = 'flask_app/'
    csv_root = 'csv_files/'
    image_root = 'images/'
    objects_root = 'objects/'

    # Retrieving metadata to assemble url locations of predicted
    # similar images
    table_cvs_file_path = root_path + csv_root + 'Table_metadata.csv'
    Table_metadata = \
    db.get_metadata_from_csv(table_cvs_file_path).transpose()

    pca_flag = 'PCA'
    ncomponents = 128
    cases = ['COLOR','CNN']
    N_neigh = 10   
    neighbor_lists = []
 
    for case in cases:
 
        if(case == 'CNN'):
          #net, transformer = cnn_caffe_features.set_caffe()
          net, transformer = get_cnn_objects()
	  ref_feature_vec = cnn_caffe_features.get_features_caffe(image_input,\
          net,transformer,'GRAY')
      	  ref_feature_vec = ref_feature_vec.reshape(1,-1) 
      	  algo_name = 'ball_tree'
      	  metric_name = 'cityblock'
    	elif(case == 'COLOR'):
      	  #ccv_ref = call_pyccv_mine(image_input)
      	  ccv_ref =  call_color_sig(image_input) 
      	  ref_feature_vec = ccv_ref[:,None].transpose()
      	  algo_name = 'brute'
      	  metric_name = None 

        if((pca_flag == 'PCA') and (case == 'CNN')):
          pca_file_name = \
          root_path + objects_root + case + '/PCA_obj_' + str(ncomponents) + '.pkl'
          pca_trans = joblib.load(pca_file_name)
          ref_feature_vec = pca_trans.transform(ref_feature_vec)

    	knn_file_name = root_path + objects_root + case + '/knn_object.pkl'
    	if(os.path.exists(knn_file_name)):
      	  print "Loading pickle file"
     	  knn_object = joblib.load(knn_file_name)    
    	else:
      	  # Retrieving feature matrix from csv file
      	  # Skipping first row (column labels)
      	  feat_csv_file_path = \
          root_path + csv_root + case + '/Feature_matrix.csv'
      	  #print os.getcwd()
      	  Feature_matrix = db.get_Feature_from_csv(feat_csv_file_path)[1:]
          if((pca_flag == 'PCA') and (case == 'CNN')):
            Feature_matrix = pca_trans.transform(Feature_matrix)
      	  knn_object = \
      	  predict.build_knn(Feature_matrix,knn_file_name,algo_name,metric_name)

        # Performing the knn predictions
        neighbor_indices = \
        predict.neighbor_indices(knn_object,ref_feature_vec,N_neigh)
        neighbor_lists.append(neighbor_indices)

    image_color_urls = []
    image_desg_urls = []
    image_rand_urls = []
    for i_loop in np.arange(0,3):
        for image_index in np.arange(0,N_neigh):
            if(i_loop == 0):
              db_index = neighbor_lists[0][0][image_index]
            elif(i_loop == 1):
              db_index = neighbor_lists[1][0][image_index]
            elif(i_loop == 2):
              db_index = np.random.randint(0,Table_metadata.shape[1])
            image_path = image_root + str(Table_metadata[db_index].iloc[1]) + \
                   '/' + str(Table_metadata[db_index].iloc[2])
            if(i_loop == 0):
               image_color_urls.append(image_path)
            elif(i_loop == 1):
               image_desg_urls.append(image_path)
            elif(i_loop == 2):
               image_rand_urls.append(image_path)   

    return {"sim_image_color_urls": image_color_urls, \
            "sim_image_desg_urls": image_desg_urls, \
            "rand_image_urls": image_rand_urls} 
    
