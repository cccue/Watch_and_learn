import numpy as np
from scipy import ndimage as ndi
from skimage.measure import moments
from skimage import transform, filters

def transform_rot(image):
    
    # Need black background (0 boundary condition) 
    # for rotational transform
    # Thats why Sobel or inverse transformation here
    #image_ref = filters.sobel(image)
    image_ref =  1.0 - image
   
    # Center of mass to be used as rotation center
    m = moments(image_ref,order=1)    
    cx = m[1, 0]/m[0, 0]
    cy = m[0, 1]/m[0, 0]
    com = cx, cy
   
    # This next step is perfect in the math but the rotation angle
    # it generates varies drastically with changes in the watch image
    # thus its not robust enough for universal alignment.
    # Therefore we add an extra rotation step after it.
    # Ascertaining rotation angle from FFT transform
    ind1 = np.arange(image.shape[0],dtype=float)
    ind2 = np.arange(image.shape[1],dtype=float)[:,None]
    angle = \
    np.angle(ind1-com[0]+1j*(ind2-com[1]))
    exp_theta = np.exp(1j*angle)
    angle_rot = np.angle(np.sum(np.sum(image_ref*exp_theta,axis=1)),deg=True)
    # Creating temporary rotated version of input image 
    image_rot_aux = \
    transform.rotate(image,angle_rot,resize=False,center=com,mode='nearest')

    # Second rotation step based on Inertia tensor
    # Again need 0 boundary condition away from object and
    # thus Sobel or inverse transform
    #image_ref = filters.sobel(image_rot_aux)
    image_ref =  1.0 - image_rot_aux

    m = moments(image_ref,order=2)
    Ixx = m[2, 0]/m[0, 0] - np.power(cx,2)
    Iyy = m[0, 2]/m[0, 0] - np.power(cy,2)
    Ixy = m[1, 1]/m[0, 0] - cx*cy
    inertia = [[Ixx, Ixy],[Ixy, Iyy]]
    w, v = np.linalg.eig(inertia)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    cross = np.cross(v[:,0],v[:,1])
    # Ensuring eigenvectors satisfy right-hand rule
    if (cross < 0):
       v[:,1] *= -1
    
    # Ascertaining rotation angle from inertia tensor eigenvectors
    angle_rad = np.arctan2(v[1,0],v[0,0]) + np.pi/2
    angle_rot = np.degrees(angle_rad)
    
    # Creating final rotated version of input image
    image_rot = \
    transform.rotate(image_rot_aux,angle_rot,resize=False,center=com,mode='nearest')

    return image_rot

def transform_trans(image):

    # Ascertaining translational shift from FFT transform    
    fft = np.fft.fftshift(np.fft.fft2(image))
    scale_x = -image.shape[0]/(2*np.pi)
    scale_y = -image.shape[1]/(2*np.pi)

    x_order = [image.shape[0]/2-1,image.shape[0]/2]
    y_order = [image.shape[1]/2,image.shape[1]/2-1]

    angle_x = np.angle(fft[x_order[0]][x_order[1]])
    angle_y = np.angle(fft[y_order[0]][y_order[1]])

    shift = scale_x*angle_x, scale_y*angle_y
   
    # Creating translated version of input image 
    image_trans = ndi.shift(image,shift,mode='wrap',prefilter='True')

    return image_trans


def set_invariant_rep(image):
    
    dims = image.shape
    offset = dims[0]/2, dims[1]/2
    skin = int(dims[0]/10), int(dims[1]/10)

    # Creating auxiliary images. They have to be initialized to 1 as
    # it is  assumed watch on input grayscale image lies on white background
    image_tmp = np.ones((2*dims[0],2*dims[1])).astype(image.dtype)
    image_rot = np.ones((dims)).astype(image.dtype)
  
    # Creating bigger version of input image to ensure rotation does not 
    # places object outside image boundaries
    image_tmp[offset[0]:dims[0]+offset[0],offset[1]:dims[1]+offset[1]] = image
    # Generating rotational invariant (RI) object
    image_rot_aux = transform_rot(image_tmp)
    # Adding translational invariance to the previously generated RI object
    image_rot_trans_aux = transform_trans(image_rot_aux)    
    # Recuperating original dimensions with extra border added
    image_rot_trans = \
    image_rot_trans_aux[offset[0]-skin[0]:dims[0]+offset[0]+skin[0],\
                        offset[1]-skin[1]:dims[1]+offset[1]+skin[1]]
    
    return image_rot_trans
