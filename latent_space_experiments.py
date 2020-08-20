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
import matplotlib.pyplot as plt

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

def draw_boxplot(data, edge_color, fill_color, labels, show=True):
    """[summary]

    Parameters
    ----------
    data : [numpy.array], dimension N x D
        Input data. N is the number of samples for each experiment, D is the number of experiments or classes
    edge_color : [str]
        Color for the box and whiskers edges. Color can be chosen between matplotlib.colors.CSS4 colors
    fill_color : [str]
        Color for the box and whiskers fill. Color can be chosen between matplotlib.colors.CSS4 colors
    labels : [list]
        List of strings of length D. Each string will be the name of a tick
    show : bool, optional
        Whether to show the plot or handle it outside this function, by default True

    Returns
    -------
    [dict]
        Returns dictionary for the boxplot
    """
    bp = plt.boxplot(data, patch_artist=True, labels=labels, showmeans=True, meanline=True, showfliers=False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color) 

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)    

    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.3)

    if show:
        plt.show()

    return bp

def compute_distance_matrix(data):
    """Compute distance between every sample and all the others

    Parameters
    ----------
    data : [numpy.array], dimension NxD
        Data matrix. N=number of samples, D=number of features

    Returns
    -------
    [numpy.array], dimension NxN
        L2 Distance matrix. Entry i,j is the distance between i-th and j-th sample 
    """

    distance_matrix = np.zeros((data.shape[0], data.shape[0]), dtype=float)

    for idx in range(data.shape[0]):
        for jdx in range(idx, data.shape[0]):
            distance_matrix[idx, jdx] = np.linalg.norm(data[idx, :] - data[jdx, :])
    
    distance_matrix += distance_matrix.transpose()

    return distance_matrix

def add_gaussian_noise(arr_noiseless, noise_mean_vector, noise_cov_matrix):
    """Add gaussian noise to a NxD array

    Parameters
    ----------
    arr_noiseless : [np.array], dimension NxD
        Array without noise
    noise_mean_vector : [np.array], dimension 1xD
        Mean vector for the gaussian noise
    noise_cov_matrix : [np.array], dimension DxD
        Covariance matrix for the gaussian noise

    Returns
    -------
    [np.array], dimension NxD
        Array with added noise, same shape as input
    """

    arr_noiseless = np.reshape(arr_noiseless, arr_noiseless.shape[-2:])

    # Check dimensions
    if arr_noiseless.shape[-1] != noise_mean_vector.shape[-1] or arr_noiseless.shape[-1] != noise_cov_matrix.shape[-1]:
        print('Array and noise shape mismatch')
        return arr_noiseless

    noise_samples = np.random.multivariate_normal(noise_mean_vector, noise_cov_matrix, arr_noiseless.shape[0])
    return arr_noiseless + noise_samples
    
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
    noise_sigma_sq = 0.0000111

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

def compute_dispersion_single_pose_noise(model_func, model_path, save_path, pcd_folder):
    # Load a bunch of undistorted point clouds, disturb each one and measure dispersion properties of their embeddings

    dispersion_measures = {}

    # Setup model
    _, get_embedding, _ , _ = get_sdf_prediction(model_func, model_path)

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
    noise_sigma_sq = 0.0000111

    noise_mean_vector = np.ones((3)) * noise_mean
    noise_cov_matrix = np.eye(3) * noise_sigma_sq

    # How many times a single point cloud should be perturbed
    number_of_experiments = 1000

    for mesh in tqdm(meshes):

        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_pcd(mesh, pcd_folder, object_frame=False, verbose=False)
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)        

        if pc_ is None:
            print(mesh, " has no point cloud.")
            continue

        # Create empty 256-column matrix to fill with embeddings
        embeddings = np.empty((number_of_experiments, 256))

        # Embed the noiseless pc
        pc_ = np.reshape(pc_, (1,1000,3))
        embedding_noiseless = get_embedding(pc_)

        for experiment_idx in range(number_of_experiments):

            # Perturb point cloud with additive gaussian noise
            pc_noisy_ = add_gaussian_noise(pc_, noise_mean_vector, noise_cov_matrix)
            point_clouds_ = np.reshape(pc_noisy_, (1,1000,3))

            # Get embedding and save it
            embedding = get_embedding(point_clouds_)

            embeddings[experiment_idx, :] = embedding

        # Compute distances between noisy embeddings and non-noisy reference
        embeddings_distance_matrix = np.linalg.norm(embeddings-embedding_noiseless, axis=1, keepdims=True) 

        embeddings_mean = np.mean(embeddings_distance_matrix)
        embeddings_variance = np.var(embeddings_distance_matrix)

        # Record results

        dispersion_measures[mesh] = (embeddings_mean, embeddings_variance, embeddings_distance_matrix)

    # Plot dispersion measures
    pc_names = []
    x = np.arange(len(dispersion_measures.keys()))
    y = np.empty(0)
    variances = np.empty(0)
    distances = np.empty((number_of_experiments, 0))
    for pc, measures in dispersion_measures.items():
        pc_names += [pc]
        y = np.append(y, measures[0])
        variances = np.append(variances, measures[1])
        distances = np.append(distances, measures[2], axis=1)
    
    # plt.errorbar(x, y, variances, linestyle='None', marker='o', ecolor='orange')
    # plt.xticks(x, pc_names, rotation=70)
    # plt.title(r'Latent space distance from noiseless embedding. Noise $\sigma^2$=' +str(noise_sigma_sq))
    # plt.subplots_adjust(bottom=0.3)

    # plt.figure()

    # Plot as a boxplot
    draw_boxplot(distances, edge_color='firebrick', fill_color='silver', labels=pc_names, show=False)
    plt.title(r'Latent space distance from noiseless embedding. Noise $\sigma^2$=' +str(noise_sigma_sq))
    # plt.plot(y, linestyle='None', marker='.', color='k')

    # Plot as histograms
    fig, ax = plt.subplots(3,7, sharey=True)
    for idx, val in enumerate(pc_names):
        ax.flat[idx].hist(distances[:, idx], color='b')
        ax.flat[idx].set_title(pc_names[idx])
    fig.suptitle(r'Latent space distance from noiseless embedding. Noise $\sigma^2$=' +str(noise_sigma_sq))

    plt.show()

    # Save experiment data to file
    import pickle
    save_dir = os.path.join(save_path, 'experiment_data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'experiment_data.pickle'), 'wb') as handle:
        pickle.dump(dispersion_measures, handle, pickle.HIGHEST_PROTOCOL)

    return dispersion_measures

def compute_dispersion_multi_pose_no_noise(model_func, model_path, save_path, pcd_folder):
    # Load a bunch of point clouds of objects rendered from different poses with no noise, and compute some measures on their embeddings
    # Point clouds must have the naming convention [obj_name]_xxxx_pc.pcd where xxxx is the pose index

    # Get names of partial views (no extension)
    import glob
    point_cloud_names = [os.path.splitext(os.path.basename(filename))[0] for filename in glob.glob(pcd_folder + "**/*.pcd")]
  
    # Setup model
    _, get_embedding, _ , _ = get_sdf_prediction(model_func, model_path)
    
    # Each point cloud will have an associated class in a len(point_cloud_names) long list
    object_labels = []

    embeddings = np.empty((0, 256))

    for pc_name in tqdm(point_cloud_names):

        # Point cloud for this view.
        pc, length, scale, centroid_diff = get_pcd(pc_name, pcd_folder, object_frame=False, verbose=False)       
        if pc is None:
            print(mesh, " has no point cloud.")
            continue

        # Embed the pc
        pc = np.reshape(pc, (1,1000,3))
        embedding = get_embedding(pc)
        embeddings = np.append(embeddings, embedding, axis=0)

        # Record pc class
        object_labels.append[pc_name[:-8]]

    distance_matrx = compute_distance_matrix(embeddings)
    
    # TODO from here

    







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

    # mesh_noisy_pc(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path=os.path.join(mesh_folder, 'meshes_noisy_pc'),
    #     pcd_folder=pcd_folder)

    compute_dispersion_single_pose_noise(
        model_func=model_func,
        model_path=model_folder,
        save_path='/home/fbottarel/workspace/PointSDF/latent_space_exp',
        pcd_folder=pcd_folder)

    
    