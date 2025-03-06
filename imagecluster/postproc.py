import os
import shutil

from matplotlib import pyplot as plt
import numpy as np

from . import calc as ic

pj = os.path.join


def plot_clusters(clusters, images, max_csize=None, mem_limit=1024**3):
    """Plot `clusters` of images in `images` with spacing between clusters.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    images : see :func:`~imagecluster.io.read_images`
    max_csize : int
        plot clusters with at most this many images
    mem_limit : float or int, bytes
        hard memory limit for the plot array
    """
    assert len(clusters) > 0, "`clusters` is empty"
    stats = ic.cluster_stats(clusters)
    if max_csize is not None:
        stats = stats[stats[:,0] <= max_csize, :]
    
    # number of clusters
    ncols = stats[:,1].sum()
    # csize (number of images per cluster)
    nrows = stats[:,0].max()
    shape = images[list(images.keys())[0]].shape[:2]
    
    # Add padding between clusters
    padding = 20  # pixels of white space between clusters
    mem = nrows * shape[0] * (ncols * shape[1] + padding * ncols) * 3
    
    if mem > mem_limit:
        raise Exception(f"size of plot array ({mem/1024**2} MiB) > mem_limit "
                        f"({mem_limit/1024**2} MiB)")
    
    # Create array with padding
    total_width = ncols * shape[1] + (ncols-1) * padding
    arr = np.ones((nrows*shape[0], total_width, 3), dtype=np.uint8) * 255
    
    icol = -1
    for csize in stats[:,0]:
        for cluster in clusters[csize]:
            icol += 1
            for irow, filename in enumerate(cluster):
                image = images[filename]
                # Add padding in x-direction
                x_start = icol * (shape[1] + padding)
                arr[irow*shape[0]:(irow+1)*shape[0],
                    x_start:x_start+shape[1], :] = image
    
    print(f"plot array ({arr.dtype}) size: {arr.nbytes/1024**2} MiB")
    fig,ax = plt.subplots(figsize=(15, 10))
    ax.imshow(arr)
    ax.axis('off')
    
    # Add cluster borders
    for i in range(ncols):
        x = i * (shape[1] + padding)
        ax.axvline(x=x-padding/2, color='red', linestyle='--', alpha=0.5)
        # Add label - try different y position
        ax.text(x + shape[1]/2, 5, f'{i+1}',  # Changed y position to 5
                horizontalalignment='center',
                verticalalignment='top',  # Added vertical alignment
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))  # Added white background
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig,ax

def visualize(*args, **kwds):
    """Interactive wrapper of :func:`plot_clusters`. Just calls ``plt.show`` at
    the end. Doesn't return ``fig,ax``.
    """
    plot_clusters(*args, **kwds)
    plt.show()


def make_links(clusters, cluster_dr):
    """In `cluster_dr`, create nested dirs with symlinks to image files
    representing `clusters`.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    cluster_dr : str
        path
    """
    print("cluster dir: {}".format(cluster_dr))
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for csize, group in clusters.items():
        for iclus, cluster in enumerate(group):
            dr = pj(cluster_dr,
                    'cluster_with_{}'.format(csize),
                    'cluster_{}'.format(iclus))
            for fn in cluster:
                link = pj(dr, os.path.basename(fn))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(os.path.abspath(fn), link)

# Adding methods for feedback system

def visualize_smart_clusters(clusters, images, feedback_system=None):
    """
    Enhanced cluster visualization with feedback info
    """
    
    # Original cluster visualization
    plot_clusters(clusters, images)

def plot_learning_progress(feedback_history):
    """
    Learning progress over time dikhata hai
    """
    if not feedback_history:
        return
        
    fig, ax = plt.subplots()
    
    times = [f['timestamp'] for f in feedback_history]
    accepted = [f['accepted'] for f in feedback_history]
    
    # Plot success rate over time
    ax.plot(times, accepted, 'g-', label='Success Rate')
    ax.set_title('System Learning Progress')
    ax.set_xlabel('Time')
    ax.set_ylabel('Success Rate')
    ax.legend()

def show_cluster_stats(feedback_system):
    """
    Clustering statistics display karta hai
    """
    stats = feedback_system.get_cluster_confidence()
    
    fig, ax = plt.subplots()
    ax.bar(['Good Clusters', 'Bad Clusters'], 
           [stats['positive_feedback'], stats['negative_feedback']])
    ax.set_title('Clustering Performance')
