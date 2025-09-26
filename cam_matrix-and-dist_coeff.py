import numpy as np
with np.load('camera_params.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coefs = data['dist_coefs']
    print("Camera matrix shape:", camera_matrix.shape)
    print("Distortion coefficients:", dist_coefs)