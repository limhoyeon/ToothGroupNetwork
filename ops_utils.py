from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from sklearn.neighbors import KDTree
import torch 
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def clustering_points(moved_points, method, num_of_clusters=None):
    """
    input:
        moved_points => type numpy => B, N, 3
        method => "DBSCAN" "aggl" "ETC",,,
        num_of_cluster => type list[int] => if selected method predefined needs num of cluster, num of clustrer have to be set.
    output:
        cluster_centroids => B, *, 3 => 3d array이고, 한 줄에 centroids 목록 쭉
        cluster_centroids_labels => B, * => 3d array이고, 각 centroid가 어떤 label인지?
        fg_points_labels_ls => B, N => 2d array이고, 각 포인트의 label이 명시됨

    """

    cluster_centroids = []
    cluster_centroids_labels = []
    fg_points_labels_ls = []
    for batch_idx in range(len(moved_points)):
        if method=="dbscan":
            clustering = DBSCAN(eps=0.03, min_samples=60).fit(moved_points[batch_idx], 3)
        elif method=="aggl":
            clustering = AgglomerativeClustering(num_of_clusters[batch_idx]).fit(moved_points[batch_idx])
        elif method=="kmeans":
            clustering = KMeans(num_of_clusters[batch_idx], init="k-means++").fit(moved_points[batch_idx])
        elif method=="mean_shift":
            clustering = MeanShift(bandwidth=0.05).fit(moved_points[batch_idx])
        else:
            clustering = GaussianMixture(n_components=num_of_clusters[batch_idx], random_state=0).fit(moved_points[batch_idx])
        unique_labels = np.unique(clustering.labels_)
        fg_points_labels_ls.append(clustering.labels_)

        batch_cluster_centroids = []
        batch_cluster_centroids_labels = []
        for label in unique_labels:
            if(label != -1):
                batch_cluster_centroids.append(np.mean(moved_points[batch_idx][clustering.labels_==label],axis=0))
                batch_cluster_centroids_labels.append(label)
        cluster_centroids.append(batch_cluster_centroids)
        cluster_centroids_labels.append(batch_cluster_centroids_labels)
    return cluster_centroids, cluster_centroids_labels, fg_points_labels_ls


def get_eg_values(points):
    if points.shape[0] < 3:
        return np.array([0,0,0])
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.explained_variance_

def find_k_kmeans(x, DEBUG=False):
    inertia_arr = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++").fit(x)
        inertia_arr.append(kmeans.inertia_)
    inertia_arr = np.array(inertia_arr)
    diff_arr = np.diff(inertia_arr)

    flag=False
    for i in range(diff_arr.shape[0]-2, 0, -1):
        if (diff_arr[i+1] - diff_arr[i])*6 < diff_arr[i] - diff_arr[i-1]:
            flag=True
            break
        else:
            continue

    if flag:
        k = i+1
    else:
        k = 1
    if DEBUG:
        plt.plot(range(inertia_arr.shape[0]), inertia_arr)
        plt.plot(range(np.diff(inertia_arr).shape[0]), np.diff(inertia_arr))
        plt.show()
        print("cluster_num is ", k)
    return k


def get_clustering_labels(moved_points, labels):
    """get cluster labels

    Args:
        moved_points (N, 3): moved points 
        labels (N, 1): labels
    """
    teeth_cond = labels != 0
    #pd_cond = probs > 0.9
    
    super_point_cond = teeth_cond #& pd_cond

    clustering = DBSCAN(eps=0.03, min_samples=30).fit(moved_points[super_point_cond, :], 3)
    clustering_labeled_moved_points = np.concatenate([moved_points[super_point_cond, :] ,clustering.labels_.reshape(-1,1)], axis=1)
    #gu.print_3d(gu.np_to_pcd_with_label(clustering_labeled_moved_points))
    clustering_labels = clustering.labels_
    core_mask = np.zeros(clustering.labels_.shape[0]).astype(bool)
    core_mask[clustering.core_sample_indices_] = True
    label_points_arr = []
    core_label_points_arr = []
    for label in np.unique(clustering.labels_):
        if label==-1:
            continue
        label_points_arr.append(clustering_labeled_moved_points[clustering_labeled_moved_points[:,3]==label,:])
        temp_mask = (core_mask) & (clustering_labeled_moved_points[:,3]==label)
        core_label_points_arr.append(clustering_labeled_moved_points[temp_mask, :])
    label_points_arr = np.array(label_points_arr, dtype="object")
    core_label_points_arr = np.array(core_label_points_arr, dtype="object")

    eg_values=  []
    for i in range(label_points_arr.shape[0]):
        eg_values.append(get_eg_values(core_label_points_arr[i][:,:3]))
    eg_values = np.array(eg_values)

    eg_values_first_axis = eg_values[:,0]
    sorted_idxes = np.argsort(-eg_values_first_axis)
    eg_values_first_axis = eg_values_first_axis[sorted_idxes]
    prb_cluster_num_ls = []
    for i in range(3):
        if eg_values_first_axis[i] / eg_values_first_axis[3:].mean() > 8:
            prb_cluster_num_ls.append(sorted_idxes[i])
    
    for idx, prb_cluster_num in enumerate(prb_cluster_num_ls):
        cluster_points = label_points_arr[prb_cluster_num][:,:]
        #cluster_num = find_k_kmeans(cluster_points[:,:3])
        kmeans = MeanShift(bandwidth=0.07).fit(cluster_points[:,:3])
        clustering_labels[clustering_labels==prb_cluster_num] = kmeans.labels_+ 100*(idx+1)
    #core_sample_indices_?
    tree = KDTree(clustering_labeled_moved_points[clustering_labels!=-1,:3], leaf_size=2)
    nn_neighbor_idxes = tree.query(clustering_labeled_moved_points[clustering_labels==-1,:3], k=10, return_distance=False)
    nn_neighbors_labels = clustering_labels[clustering_labels!=-1][nn_neighbor_idxes]
    
    mod_labels = []
    for i in range(nn_neighbors_labels.shape[0]):
        u, c = np.unique(nn_neighbors_labels[i], return_counts=True)
        mod_labels.append(u[np.argmax(c)])
    clustering_labels[clustering_labels==-1] = np.array(mod_labels) 

    return clustering_labels

def get_nearest_neighbor_idx(org_xyz, sampled_clusters, crop_num=4096):
    """
    Input:
        org_xyz => type np => B, N, 3
        sampled_clusters => type np => B, cluster_num, 3
    Output:
        return - B, cluster_num, 4096
    """
    cropped_all = []
    for batch_idx in range(org_xyz.shape[0]):
        cropped_points = []

        tree = KDTree(org_xyz[batch_idx,:,:], leaf_size=2)
        indexs = tree.query(sampled_clusters[batch_idx], k=crop_num, return_distance = False)
        cropped_all.append(indexs)
    return cropped_all


def centering_object(points):
    points = points.permute(0,2,1)
    for point in points:
        point[:,:3] = point[:,:3] - torch.mean(point[:, :3], dim=0)
    points = points.permute(0,2,1)
    return points

def seg_label_to_cent(gt_coords, gt_seg_label):
    gt_coords = gt_coords.permute(0,2,1)
    gt_coords = gt_coords.view(-1,3)
    gt_seg_label = gt_seg_label.view(-1)

    gt_cent_coords = []
    gt_cent_exists = []
    for class_idx in range(0, 16):
        cls_cond = gt_seg_label==class_idx
        
        cls_sample_xyz = gt_coords[cls_cond, :]
        if cls_sample_xyz.shape[0]==0:
            gt_cent_coords.append(torch.from_numpy(np.array([-10,-10,-10])))
            gt_cent_exists.append(torch.zeros(1))
        else:
            centroid = torch.mean(cls_sample_xyz, axis=0)
            gt_cent_coords.append(centroid)
            gt_cent_exists.append(torch.ones(1))

    gt_cent_coords = torch.stack(gt_cent_coords)
    gt_cent_coords = gt_cent_coords.view(1, *gt_cent_coords.shape)
    gt_cent_coords = gt_cent_coords.permute(0,2,1)
    gt_cent_exists = torch.stack(gt_cent_exists)
    gt_cent_exists = gt_cent_exists.view(1, -1)

    return gt_cent_coords, gt_cent_exists

def get_indexed_features(features, cropped_indexes):
    """
    Input:
        features => type torch cuda/np => B, channel, N
        cropped indexes => type torch cuda/np => B, cluster_num, 4096
    Output:
        cropped_item_ls => type torch cuda/np => new batch B, channel, 4096
    """
    cropped_item_ls = []
    for b_idx in range(len(cropped_indexes)):
        for cluster_idx in range(len(cropped_indexes[b_idx])):
            #cropped_point = torch.index_select(features[b_idx,:,:], 1, torch.tensor(cropped_indexes[b_idx][cluster_idx]).cuda())
            cropped_point = features[b_idx][:, cropped_indexes[b_idx][cluster_idx]]
            cropped_item_ls.append(cropped_point)
    if type(cropped_item_ls[0]) == torch.Tensor:
        cropped_item_ls = torch.stack(cropped_item_ls, dim=0)
    elif type(cropped_item_ls[0]) == np.ndarray:
        cropped_item_ls = np.stack(cropped_item_ls, axis=0)
    else:
        raise "someting unknwon type"
    return cropped_item_ls

