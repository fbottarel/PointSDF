# Code to experiment with the PointSDF latent space

# Get PointConv model functions
from sdf_pointconv_model import get_pointconv_model, get_sdf_model, get_embedding_model, get_sdf_prediction
from run_sdf_model import run_sdf
from mise import *

# Accessory packages 
import os
import sys
import numpy as np
from tqdm import tqdm

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

def mesh_objects_two_steps(model_func, model_path, save_path, pcd_folder):
    # Do the same thing as mise.mesh_objects, but extracting and saving the point cloud embedding

    # Setup model.
    _, get_embedding, _ , get_sdf_from_embedding = get_sdf_prediction(model_func, model_path)

    # Get names of partial views.
    import glob
    meshes = [os.path.splitext(os.path.basename(filename))[0] for filename in glob.glob(pcd_folder + "**/*.pcd")]
    
    # Bounds of 3D space to evaluate in: [-bound, bound] in each dim.
    bound = 0.8
    # Starting voxel resolution.
    initial_voxel_resolution = 16
    # Final voxel resolution.
    final_voxel_resolution = 32

    # Mesh the views.
    for mesh in tqdm(meshes):

        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_pcd(mesh, pcd_folder, object_frame=False, verbose=False)
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)    

        if pc_ is None:
            print(mesh, " has no point cloud.")
            continue
        point_clouds_ = np.reshape(pc_, (1,1000,3))

        embedding = get_embedding(point_clouds_)

        embedding_filename = os.path.join(save_path, mesh + '_embedding.npy')
        np.save(embedding_filename, embedding)

        def get_sdf_embedding_query(query_points):
            return get_sdf_from_embedding(embedding, query_points)

        mise_voxel(get_sdf_embedding_query, bound, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff, os.path.join(save_path, mesh + '.obj'), verbose=False)

    return
    
def mesh_disturb_embedding(model_func, model_path, save_path, pcd_folder):
    # Load point cloud and embed it, disturb it with gaussian noise and obtain mesh completions based on it
    
    # Setup model
    _, get_embedding, _ , get_sdf_from_embedding = get_sdf_prediction(model_func, model_path)

    # Get names of partial views.
    import glob
    meshes = [os.path.splitext(os.path.basename(filename))[0] for filename in glob.glob(pcd_folder + "**/*.pcd")]
    
    # Bounds of 3D space to evaluate in: [-bound, bound] in each dim.
    bound = 0.8
    # Starting voxel resolution.
    initial_voxel_resolution = 16
    # Final voxel resolution.
    final_voxel_resolution = 64

    number_of_experiments = 5

    for mesh in tqdm(meshes):

        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_pcd(mesh, pcd_folder, object_frame=False, verbose=False)
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)    

        if pc_ is None:
            print(mesh, " has no point cloud.")
            continue
        point_clouds_ = np.reshape(pc_, (1,1000,3))

        embedding = get_embedding(point_clouds_)

        for experiment_idx in range(number_of_experiments):

            noise = np.random.normal(0,1, size=embedding.shape)

            embedding_exp = embedding + noise

            def get_sdf_embedding_query(query_points):
                return get_sdf_from_embedding(embedding_exp, query_points)

            mise_voxel(get_sdf_embedding_query, bound, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff, os.path.join(save_path, mesh + "_" + str(experiment_idx) + '.obj'), verbose=False)


if __name__ == "__main__":
    
    # Proceed to complete and mesh point clouds without further ado
    # mesh_objects(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path=mesh_folder,
    #     pcd_folder=pcd_folder)

    # mesh_objects_two_steps(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path=mesh_folder,
    #     pcd_folder=pcd_folder)

    mesh_disturb_embedding(
        model_func=model_func,
        model_path=model_folder,
        save_path=os.path.join(mesh_folder, 'noisy_meshes'),
        pcd_folder=pcd_folder)

    
    
