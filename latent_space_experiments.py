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
import collections

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

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    import matplotlib

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def heatmap2(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    ax.imshow(data)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(top=False, bottom=True,
                labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="right",
            rotation_mode="anchor")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, '%.1f'%(data[i, j]), ha="center", va="center", color="w")

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title('Distance matrix')

    return im

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
    # As input, one point cloud per object

    dispersion_measures = {}
    noiseless_embeddings = np.empty((0,256), float)

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
    number_of_experiments = 100

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
        noiseless_embeddings = np.append(noiseless_embeddings, embedding_noiseless, axis=0)

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

def compute_dispersion_multi_class_multi_pose_no_noise(model_func, model_path, save_path, pcd_folder):
    # Load point clouds of different objects rendered from different poses (no noise added), and compute their dispersion wrt their mean embedding
    # Point clouds must have the naming convention [obj_name]_xxxx_pc.pcd, where [xxxx] stands for the pose

    # Get names of partial views (no extension)
    import glob
    point_cloud_names = [os.path.splitext(os.path.basename(filename))[0] for filename in glob.glob(pcd_folder + "**/*.pcd")]

    # Setup model
    _, get_embedding, _ , _ = get_sdf_prediction(model_func, model_path)

    # Each point cloud will have an associated class in a len(point_cloud_names) long list
    object_labels = []

    embeddings = np.empty((0, 256))

    for pc_name in tqdm(sorted(point_cloud_names)):

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
        object_labels.append(pc_name[:-8])

    # We want all the same objects' embeddings together in order to plot them
    embeddings_by_class = collections.OrderedDict()

    for idx_label, label  in enumerate(object_labels):
        if label not in embeddings_by_class.keys():
            # Create dict entry if not present
            embeddings_by_class[label] = np.empty((0, 256))

        embeddings_by_class[label] = np.append(embeddings_by_class[label], np.reshape(embeddings[idx_label, :], (1,256)), axis=0)

    # Compute the mean embedding for all the poses wrt the same object
    # Create a NxD object, where D is the number of classes and N the number of samples per class
    # Save the mean embedding for each class in a MxF object, where M is the number of classes and F the number of features
    distances_from_mean = np.empty((embeddings.shape[0]/len(embeddings_by_class.keys()),0), dtype=float)
    embeddings_means = np.empty((0, 256), float)

    for obj_name, obj_embeddings in embeddings_by_class.items():

        # For every object, compute the mean embedding and distance from the mean embedding of each sample
        embeddings_mean = np.mean(obj_embeddings, axis=0, keepdims=True)
        tmp = np.linalg.norm(embeddings_by_class[obj_name]-embeddings_mean, axis=1, keepdims=True)
        distances_from_mean = np.append(distances_from_mean, tmp, axis=1)
        embeddings_means = np.append(embeddings_means, embeddings_mean, axis=0)

    # Draw the box plot
    fig1, ax1 = plt.subplots()
    draw_boxplot(distances_from_mean, edge_color='firebrick', fill_color='silver', labels=embeddings_by_class.keys(), show=False)

    ax1.set_title("Infra-class distribution with respect to mean embedding")
    fig1.tight_layout()

    # Draw heatmap of distances
    inter_class_distance_matrix = compute_distance_matrix(embeddings_means)

    fig2, ax2 = plt.subplots()
    im, cbar = heatmap(inter_class_distance_matrix,
                        embeddings_by_class.keys(),
                        embeddings_by_class.keys(),
                        ax=ax2,
                        cmap="YlGn",
                        cbarlabel="L2 Distance in latent space"
                        )

    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    ax2.set_title("L2 inter-class latent space distance between the mean mapping of different poses")
    fig2.tight_layout()

    plt.subplots_adjust(bottom=0.17, right=0.86)

    # plt.show()

    # TODO: DOCUMENT THIS
    from sklearn import manifold
    import matplotlib.cm as cm

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embeddings)


    hues = cm.rainbow(np.linspace(0,1, len(embeddings_by_class.keys())))
    tsne_colors = []
    tsne_series_name = []
    for h in hues:
        tsne_colors += [h]*(embeddings.shape[0]/len(embeddings_by_class.keys()))

    samples_per_class = embeddings.shape[0]/len(embeddings_by_class.keys())

    fig3, ax3 = plt.subplots()

    for idx, h in enumerate(hues):
        ax3.scatter(X_tsne[idx*samples_per_class:(idx+1)*samples_per_class, 0], X_tsne[idx*samples_per_class:(idx+1)*samples_per_class, 1], color=h, label=embeddings_by_class.keys()[idx], marker=r"${}$".format(idx), s=80)

    ax3.legend()
    ax3.set_title("t-SNE (2-dimensional) of embedding vectors")
    plt.show()



    return




















def compute_inter_class_distance_single_pose_no_noise(model_func, model_path, save_path, pcd_folder):
    # Load a bunch of point clouds of objects rendered from a single with no noise, and compute some measures on their embeddings
    # Point clouds must have the naming convention [obj_name]_pc.pcd

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
        object_labels.append(pc_name[:-7])

    distance_matrix = compute_distance_matrix(embeddings)

    # Draw the distance matrix
    # fig, ax = plt.subplots()
    # ax.matshow(distance_matrix)
    # ax.set_xticks(np.arange(distance_matrix.shape[1]))
    # ax.set_yticks(np.arange(distance_matrix.shape[0]))
    # ax.set_xticklabels(object_labels)
    # ax.set_yticklabels(object_labels)
    # ax.xaxis.set_ticks_position('bottom')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    # plt.setp(ax.get_yticklabels(), ha="right",
    #             rotation_mode="anchor")

    # for i in range(distance_matrix.shape[0]):
    #     for j in range(distance_matrix.shape[1]):
    #         text = ax.text(j, i, '%.1f'%(distance_matrix[i, j]), ha="center", va="center", color="w")

    # ax.set_title('Distance matrix')

    fig, ax = plt.subplots()

    im, cbar = heatmap(distance_matrix, object_labels, object_labels, ax=ax,  cmap="YlGn", cbarlabel="L2 Distance in latent space")

    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    ax.set_title('Distance matrix')
    fig.tight_layout()
    plt.show()

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

    # compute_dispersion_single_pose_noise(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path='/home/fbottarel/workspace/PointSDF/latent_space_exp',
    #     pcd_folder='/home/fbottarel/workspace/PointSDF/pcs/rendered_ycb/processed_pcs')

    # compute_inter_class_distance_single_pose_no_noise(
    #     model_func=model_func,
    #     model_path=model_folder,
    #     save_path='/home/fbottarel/workspace/PointSDF/latent_space_exp',
    #     pcd_folder='/home/fbottarel/workspace/PointSDF/pcs/rendered_ycb/processed_pcs'
    # )

    compute_dispersion_multi_class_multi_pose_no_noise(
        model_func=model_func,
        model_path=model_folder,
        save_path='/home/fbottarel/workspace/PointSDF/latent_space_exp',
        pcd_folder='/home/fbottarel/workspace/PointSDF/pcs/rendered_ycb_multiview/processed_pointclouds'
    )
