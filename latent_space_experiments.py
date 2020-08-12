# Code to experiment with the PointSDF latent space

import os
import sys

# Get PointConv model functions
from sdf_pointconv_model import get_pointconv_model
from run_sdf_model import run_sdf
from mise import mesh_objects, mesh_objects_two_steps

# Hardcoded variables, because why not
pcd_folder = '/home/fbottarel/workspace/PointSDF/pcs/pointclouds_435/processed_pointclouds'
model_func = get_pointconv_model
model_name = 'pointconv_mse_cf'
model_path = '/home/fbottarel/workspace/PointSDF/models/reconstruction'
log_path   = '/home/fbottarel/workspace/PointSDF/models/reconstruction'
mesh_folder= '/home/fbottarel/workspace/PointSDF/meshes/'

model_folder = os.path.join(model_path, model_name)

# Needed PointConv path
os.environ['POINTCONV_HOME'] = '/home/fbottarel/workspace/pointconv'
sys.path.append(os.environ['POINTCONV_HOME'])

# mesh_objects(
#     model_func=model_func,
#     model_path=model_folder,
#     save_path=mesh_folder,
#     pcd_folder=pcd_folder)


mesh_objects_two_steps(
    model_func=model_func,
    model_path=model_folder,
    save_path=mesh_folder,
    pcd_folder=pcd_folder)
