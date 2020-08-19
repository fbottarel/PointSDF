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
pcd_folder = '/home/fbottarel/workspace/PointSDF/pcs/rendered_ycb/processed_pcs'
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

def mesh_noisy_pc(model_func, model_path, save_path, pcd_folder):
    # Load point cloud, disturb it with some noise distribution and obtain embeddings + completions on it

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

    # Parameters of the noise distribution
    noise_mean = 0
    noise_sigma_sq = 0.005

    noise_mean_vector = np.ones((3)) * noise_mean
    noise_cov_matrix = np.eye(3) * noise_sigma_sq

    # How many times a single point cloud should be perturbed
    number_of_experiments = 5

    for mesh in tqdm(meshes):

        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_pcd(mesh, pcd_folder, object_frame=False, verbose=False)
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)        

        if pc_ is None:
            print(mesh, " has no point cloud.")
            continue

        for experiment_idx in range(number_of_experiments):

            # Perturb point cloud with additive gaussian noise
            noise_samples = np.random.multivariate_normal(noise_mean_vector, noise_cov_matrix, pc_.shape[0])

            if noise_samples.shape != pc_.shape:
                print(mesh, " dimensions are not compatible with noise sample matrix.")
                continue

            pc_noisy_ = pc_ + noise_samples
            point_clouds_ = np.reshape(pc_, (1,1000,3))

            # Get embedding and save it
            embedding = get_embedding(pc_noisy_)
            embedding_filename = os.path.join(save_path, mesh + str(experiment_idx) + '_noisy_pc_embedding.npy')
            np.save(embedding_filename, embedding)

            def get_sdf_embedding_query(query_points):
                return get_sdf_from_embedding(embedding, query_points)

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

    # mesh_disturb_embedding(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path=os.path.join(mesh_folder, 'noisy_meshes'),
    #     pcd_folder=pcd_folder)

    mesh_noisy_pc(
        model_func=model_func,
        model_path=model_folder,
        save_path=os.path.join(mesh_folder, 'meshes_noisy_pc'),
        pcd_folder=pcd_folder)
