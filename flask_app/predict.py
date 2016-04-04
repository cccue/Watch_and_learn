import numpy as np
from sklearn.neighbors import NearestNeighbors
from emd import emd
from sklearn.externals import joblib

# Earth's movers distance on signatures from
# color quantization
def metric_emd_sig(vec1,vec2):

    ndim = 4
    nclusters = vec1.shape[0]/ndim

    vec1 = vec1.reshape(ndim*nclusters,1)
    vec2 = vec2.reshape(ndim*nclusters,1)

    sig1 = vec1.reshape((ndim,nclusters)).T
    sig2 = vec2.reshape((ndim,nclusters)).T

    X_weights = np.empty((nclusters,1))
    Y_weights = np.empty((nclusters,1))
    X = np.empty((nclusters,ndim-1))
    Y = np.empty((nclusters,ndim-1))

    X_weights[:,0] = sig1[:,0]
    Y_weights[:,0] = sig2[:,0]

    X[:,0:ndim-1] = sig1[:,1:ndim]
    Y[:,0:ndim-1] = sig2[:,1:ndim]

    # Next lines remove biggest cluster. Assumes it represents the
    # background
    max_index = np.argmax(X_weights)
    X_weights_red = np.delete(X_weights,max_index,axis=0)
    X_weights_red = X_weights_red/np.sum(X_weights_red,axis=0)
    X_red = np.delete(X,max_index,axis=0)

    max_index = np.argmax(Y_weights)
    Y_weights_red = np.delete(Y_weights,max_index,axis=0)
    Y_weights_red = Y_weights_red/np.sum(Y_weights_red,axis=0)
    Y_red = np.delete(Y,max_index,axis=0)

    distance = emd(X_red,Y_red,X_weights_red,Y_weights_red,distance='euclidean')
    #distance = emd(X,Y,X_weights,Y_weights,distance='euclidean') 

    return distance

# Cosine similarity
def metric_cosine(vec1,vec2):

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    distance = 1.0 - np.inner(vec1,vec2)/(norm1*norm2)
 
    return distance

# Chi-squared metric
def metric_chi_squared(vec1,vec2):

    frac = 0.5*np.power(vec1-vec2,2)/(vec1+vec2)
    distance = np.sum(frac[np.isfinite(frac)])

    return distance

# Histogram intersection metric
def metric_hist_inter(vec1,vec2):

    distance = \
    1 - np.sum(np.minimum(vec1,vec2).astype('float'))/np.sum(vec1)

    return distance

# Building the knn Object to speed up CCV similarity searches
def build_knn(feature_matrix,file_name,algo_name,metric_name):

   if(metric_name == None): metric_name = metric_emd_sig

   # Default neighbors and radius for built 
   k_neigh = 10
   radius = 10.5
   #knn_object = \
   #NearestNeighbors(k_neigh,radius,metric=metric_line)
   knn_object = \
   NearestNeighbors(k_neigh,radius,algorithm=algo_name,metric=metric_name)
   #knn_object = \
   #NearestNeighbors(k_neigh,radius,algorithm='brute',metric=metric_emd_sig) 
   knn_object.fit(feature_matrix)
   joblib.dump(knn_object,file_name,compress=1) 

   return knn_object

# Predicting indices of first N_neigh neighors in feature space 
def neighbor_indices(knn_object,feature_vec,N_neigh):

    neighbor_indices = \
    knn_object.kneighbors(feature_vec,N_neigh,return_distance=False)

    return neighbor_indices

