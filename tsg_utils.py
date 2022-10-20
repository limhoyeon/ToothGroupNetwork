from sklearn.cluster import DBSCAN, SpectralClustering, AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.neighbors import KDTree
import torch 
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
import gen_utils
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gen_utils as gu

def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def get_eg_values(points):
    if points.shape[0] < 3:
        return np.array([0,0,0])
    from sklearn.decomposition import PCA
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

def propagate_unlabeled_points(moved_points, labels):
    #labels에는 0은 잇몸, 12345678990,,,, 치아, -1이 unlabeled
    clustering_labels = labels.copy()
    #잇몸 빼고, label 된 애들을 tree 로 가져옴
    tree = KDTree(moved_points[clustering_labels>0,:3], leaf_size=2)
    nn_neighbor_idxes = tree.query(moved_points[clustering_labels==-1,:3], k=10, return_distance=False)
    nn_neighbors_labels = clustering_labels[clustering_labels>0][nn_neighbor_idxes]
    
    mod_labels = []
    for i in range(nn_neighbors_labels.shape[0]):
        u, c = np.unique(nn_neighbors_labels[i], return_counts=True)
        mod_labels.append(u[np.argmax(c)])
    clustering_labels[clustering_labels==-1] = np.array(mod_labels) 

    if False:
        clustering_labeled_moved_points[:,3] = clustering_labels
        gu.print_3d(gu.np_to_pcd_with_label(clustering_labeled_moved_points))
    return clustering_labels


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

    if False:
        clustering_labeled_moved_points[:,3] = clustering_labels
        gu.print_3d(gu.np_to_pcd_with_label(clustering_labeled_moved_points))
    return clustering_labels

def get_cluster_number(moved_points, labels, probs, DEBUG=False):
    """get cluster number

    Args:
        moved_points (N, 3): moved points 
        labels (N, 1): labels
        probs (N, 1): probs
    """
    teeth_cond = labels != 0
    #pd_cond = probs > 0.9
    
    super_point_cond = teeth_cond #& pd_cond

    clustering = DBSCAN(eps=0.03, min_samples=30).fit(moved_points[super_point_cond, :], 3)
    clustering_labeled_moved_points = np.concatenate([moved_points[super_point_cond, :] ,clustering.labels_.reshape(-1,1)], axis=1)

    label_points_arr = []
    for label in np.unique(clustering.labels_):
        if label==-1:
            continue
        label_points_arr.append(clustering_labeled_moved_points[clustering_labeled_moved_points[:,3]==label,:])
    label_points_arr = np.array(label_points_arr)

    eg_values=  []
    for i in range(label_points_arr.shape[0]):
        eg_values.append(get_eg_values(label_points_arr[i][:,:3]))
    eg_values = np.array(eg_values)

    eg_values_first_axis = eg_values[:,0]
    sorted_idxes = np.argsort(-eg_values_first_axis)
    eg_values_first_axis = eg_values_first_axis[sorted_idxes]
    prb_cluster_num_ls = []
    for i in range(3):
        if eg_values_first_axis[i] / eg_values_first_axis[3:].mean() > 13:
            prb_cluster_num_ls.append(sorted_idxes[i])

    cluster_num = len(label_points_arr)
    for prb_cluster_num in prb_cluster_num_ls:
        cluster_num -= 1
        cluster_num += find_k_kmeans(label_points_arr[prb_cluster_num][:,:3])
    if probs != cluster_num:
        a=1
        #import gen_utils as gu
        #gu.print_3d(gu.np_to_pcd_with_label(clustering_labeled_moved_points))

        #plt.scatter(range(eg_values.shape[0]), eg_values[:,0])
        #plt.show()

    return cluster_num, eg_values_first_axis[[0,1,2]] / eg_values_first_axis[3:].mean()


def dbscan_moved_points(moved_points, pred_dist):
    #moved_points = B, N, C
    #pred_dist = B,N
    #output = B, N, C
    pred_centroids = moved_points.cpu().detach().numpy()
    pred_dist = pred_dist.cpu().detach().numpy()


    sampled_centroids = []
    for batch_idx in range(pred_centroids.shape[0]):
        batch_pred_centroids = pred_centroids[batch_idx,:,:]
        batch_pred_dist = pred_dist[batch_idx, :]
        batch_pred_centroids = batch_pred_centroids[(batch_pred_dist<0.015).reshape(-1)]
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(batch_pred_centroids,3)
        unique_labels = np.unique(clustering.labels_)
        clustering_ls = []
        for label in unique_labels:
            if(label != -1):
                clustering_ls.append(np.mean(batch_pred_centroids[clustering.labels_==label],axis=0))
        sampled_centroids.append(clustering_ls)
    return sampled_centroids

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

def get_adaptive_nearest_neighbor_idx(org_xyz, sampled_clusters, crop_num=4096):
    #org_xyz -> B, 3, N
    #sampled_clusters -> B, cluster_num, 3
    #return - B, cluster_num, 4096 
    org_xyz = org_xyz.permute(0,2,1).cpu().detach().numpy()
    cropped_all = []
    for batch_idx in range(org_xyz.shape[0]):
        cropped_points = []

        tree = KDTree(org_xyz[batch_idx,:,:], leaf_size=2)
        for tooth_class_num, cluster_centroid in enumerate(sampled_clusters[batch_idx]):
            if tooth_class_num % 8 <= 4:
                cls_crop_num = crop_num*2//3
            else:
                cls_crop_num = crop_num
            indexs = tree.query([cluster_centroid], k=cls_crop_num, return_distance = False)
            cropped_points.append(indexs[0])
        cropped_all.append(cropped_points)
    return cropped_all

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


def get_nearest_neighbor_idx_radius(org_xyz, sampled_clusters, radius):
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
        indexs = tree.query_radius(sampled_clusters[batch_idx], r=radius, return_distance = False, count_only = False)
        cropped_all.append(indexs)
    return cropped_all


def get_cylinder_nearest_neighbor_idx(org_xyz, sampled_clusters, gt_centroid_exists):
    #                  0    1     2     3     4     5     6    7     8     9      10      11     12      13      14      15
    neighbor_set = [[8,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6], [0,9],[8,10],[9,11],[10,12],[11,13],[12,14],[13,15],[14]]
    #org_xyz -> B, 3, N
    #sampled_clusters -> B, cluster_num, 3
    #return - B, cluster_num, 4096 
    org_xyz = org_xyz.permute(0,2,1).cpu().detach().numpy()
    
    cropped_all = []
    for batch_idx in range(org_xyz.shape[0]):
        cropped_points = []

        #z축 align이라고 가정할 수 있는 상황에서 cylinder화를 시켜버리는 것
        tree = KDTree(org_xyz[batch_idx,:,:2], leaf_size=2)
        for tooth_class_num, cluster_centroid in enumerate(sampled_clusters[batch_idx]):
            if tooth_class_num % 8 <= 5:
                rad = 0.1
                for near_neighbor_idx in neighbor_set[tooth_class_num]:
                    if gt_centroid_exists[batch_idx, near_neighbor_idx] == 0:
                        continue
                    rad = min(rad, np.linalg.norm(sampled_clusters[batch_idx][near_neighbor_idx][:2]-cluster_centroid[:2])+0.025)
            else:
                rad = 0.2
                for near_neighbor_idx in neighbor_set[tooth_class_num]:
                    if gt_centroid_exists[batch_idx, near_neighbor_idx] == 0:
                        continue
                    rad = min(rad, np.linalg.norm(sampled_clusters[batch_idx][near_neighbor_idx][:2]-cluster_centroid[:2])+0.065)
            """
            rad=100
            basis = None
            for near_neighbor_idx in neighbor_set[tooth_class_num]:
                print(near_neighbor_idx)
                if isinstance(basis, (np.ndarray, np.generic)):
                    basis = basis-(sampled_clusters[batch_idx][near_neighbor_idx]-cluster_centroid)
                else:
                    basis = sampled_clusters[batch_idx][near_neighbor_idx]-cluster_centroid
                rad = min(rad, np.linalg.norm(sampled_clusters[batch_idx][near_neighbor_idx][:2]-cluster_centroid[:2])/2)
            
            if isinstance(basis, (np.ndarray, np.generic)):
                basis = basis/np.linalg.norm(basis)
                basis_normal = np.cross(basis, np.array([0,0,1]))
                basis = basis[:2].reshape(2,1)
                basis_normal = (basis_normal[:2]*1/2).reshape(2,1)
                new_basis = np.concatenate((basis, basis_normal),axis=1)
                transformed_org_xyz = np.matmul(new_basis, org_xyz[batch_idx,:,:2].T).T
                tree = KDTree(transformed_org_xyz[:,:2], leaf_size=2)
                indexs = tree.query_radius([np.matmul(new_basis, cluster_centroid[:2].T).T], r=rad, count_only=False, return_distance = False)
                cropped_points.append(indexs[0])
            else:
                tree = KDTree(org_xyz[batch_idx,:,:2], leaf_size=2)
                indexs = tree.query_radius([cluster_centroid[:2]], r=rad, count_only=False, return_distance = False)
                cropped_points.append(indexs[0])
                """
            indexs = tree.query_radius([cluster_centroid[:2]], r=rad, count_only=False, return_distance = False)
            cropped_points.append(indexs[0])
        cropped_all.append(cropped_points)
    return cropped_all

def get_indexed_teeth_mask(teeth_mask, cropped_indexes):
    """
    Input:
        teeth_mask => type torch cuda, boolean type => B, N(24000, full view)
        cropped indexes => type torch cuda => B, cluster_num, 4096
    Output:
        cropped_teeth_mask_ls => type torch cuda => 2d array new batch B, 임의,, 
    """
    cropped_teeth_mask_ls = []
    for b_idx in range(len(cropped_indexes)):
        for cluster_idx in range(len(cropped_indexes[b_idx])):
            teeth_mask = teeth_mask[b_idx][cropped_indexes[b_idx][cluster_idx]]
            cropped_teeth_mask_ls.append(torch.where(teeth_mask!=0))
    return cropped_teeth_mask_ls

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

def get_cluster_teeth_mask(points_labels_ls, nn_cropped_indexes, cluster_centroids_labels, gt_seg_label):
    """
    Input:
        points_labels_ls => B(full 기준), N(전체 포인트)
        nn_crop_indexes => B(full 기준), cluster 개수, 4096(인덱스들)
        cluster_centroids_labels => B(full 기준), cluster 개수
        gt_seg_label => B(full 기준), 1, N(모든 포인트)
    Output:
        cluster_teeth_mask_ls => B(new batch), 4096(bool)
        cluster_tooth_gt_num_ls => B(new batch), 1(tooth num, int)
    """
    idx = 0
    cluster_teeth_mask_ls = []
    cluster_tooth_gt_num_ls = []

    for b_idx in range(len(nn_cropped_indexes)):
        for cluster_idx in range(len(cluster_centroids_labels[b_idx])):
            #cluster num을 의미
            label = cluster_centroids_labels[b_idx][cluster_idx]

            cropped_indexes = nn_cropped_indexes[b_idx][cluster_idx]
            cropped_labels = points_labels_ls[b_idx][:, cropped_indexes].reshape(-1)
            cluster_teeth_mask = cropped_labels == label # mask, 4096 1d bool
            cropped_tooth_num_labeled_gt_points = gt_seg_label[b_idx, 0][cropped_indexes][cluster_teeth_mask]
            
            cluster_teeth_mask_ls.append(cluster_teeth_mask)

            if len(cropped_tooth_num_labeled_gt_points) > 20:
                values, counts = np.unique(cropped_tooth_num_labeled_gt_points, return_counts=True)
                gt_tooth_num = values[np.argmax(counts)]
                cluster_tooth_gt_num_ls.append([gt_tooth_num])
            else:
                cluster_tooth_gt_num_ls.append([-1])

    return cluster_teeth_mask_ls, cluster_tooth_gt_num_ls

def get_cylinder_nearest_neighbor_points_per_centroids(feature_xyz, cropped_all_indexes, sampled_db_scan):
    #org_xyz -> batch, features, N
    #cropped_all_indexes -> batch, centroid_num, 4096
    #sampled_clusters -> batch, cluster_num, 3
    #return -> batch, centroid_num, features, 4096 // centroid_num, 3, 1

    cropped_points_cr_batch = []
    for b_idx in range(len(cropped_all_indexes)):
        cropped_points = []
        for cluster_idx in range(len(cropped_all_indexes[0])):
            cropped_point = torch.index_select(feature_xyz[b_idx,:,:], 1, torch.tensor(cropped_all_indexes[b_idx][cluster_idx]).cuda())

            cropped_points.append(cropped_point)
        cropped_points_cr_batch.append(cropped_points)
    return cropped_points_cr_batch
    
def get_nearest_neighbor_points_hold_batch(feature_xyz, cropped_all_indexes, sampled_db_scan, rand_cluster_indexes):
    #org_xyz -> B, features, N
    #cropped_all_indexes -> B, centroid_num, 4096
    #sampled_clusters -> B, cluster_num, 3
    #cluster_idx -> B : int
    #random으로 하나의 teeth만 고른다.
    #return -> B, features, 4096 // B, 3, 1
    #돌아가면서 for문으로 빼는 방식으로 만들 것,
    cropped_points = []
    sampled_clusters_seleceted = []
    for b_idx in range(len(cropped_all_indexes)):
        cropped_point = torch.index_select(feature_xyz[0,:,:], 1, torch.tensor(cropped_all_indexes[b_idx][rand_cluster_indexes[b_idx]]).cuda())
        cropped_points.append(cropped_point)
        sampled_clusters_seleceted.append(sampled_db_scan[b_idx][rand_cluster_indexes[b_idx]])
    cropped_points = torch.stack(cropped_points, dim=0)
    sampled_clusters_cuda = torch.from_numpy(np.array(sampled_clusters_seleceted)).cuda()
    sampled_clusters_cuda = sampled_clusters_cuda.view((-1,3,1))
    return cropped_points, sampled_clusters_cuda
        
def concat_seg_input(cropped_features, cropped_coord, sampled_centroid_points_cuda):
    # cropped_features : B(crooped 개수), 16, 4096
    # cropped_coord : B(crooped 개수), 3, 4096
    # sampled_centroid_points_cuda : B, 3, 1
    cropped_coord = cropped_coord.permute(0,2,1)
    sampled_centroid_points_cuda = sampled_centroid_points_cuda.permute(0,2,1)
    
    ddf = square_distance(cropped_coord, sampled_centroid_points_cuda)
    ddf *= (-4)
    ddf = torch.exp(ddf)

    cropped_coord = cropped_coord.permute(0,2,1)
    ddf = ddf.permute(0,2,1)
    concat_result = torch.cat([cropped_coord, cropped_features, ddf], dim=1)
    return concat_result

def get_gt_labels_maximum(gt_label):
    # gt_label: batch_size, , cropped
    # gt_bin_label: batch_size, 1, cropped
    gt_max_labels = []
    proposal_gt_label = gt_label.view(gt_label.shape[0], -1)
    for batch_idx in range(proposal_gt_label.shape[0]):
        label_indexes, counts = proposal_gt_label[batch_idx, :].unique(sorted=True, return_counts = True)
        max_index = torch.argmax(counts[1:])
        max_label = torch.index_select(label_indexes[1:], 0, max_index)
        gt_max_labels += [max_label]
    gt_max_labels = torch.stack(gt_max_labels, dim=0)
    gt_max_labels = gt_max_labels.view(-1,1)
    
    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.zeros_like(gt_label)
    gt_bin_label[gt_label == gt_max_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    
    return gt_bin_label

def get_cluster_gt_id_by_nearest_gt_centroid(gt_centorids_coords, gt_centroids_ids, pred_centroid_coords):
    # gt_centorids_coords batch_size, 3, num of gt centroid, 모든 클러스터에 대해 , 배치마다!
    # gt_centroids_ids batch_size, 1, num of gt centroid, 모든 클러스터에 대해 , 배치마다!
    # pred_centroid_coords batch_size, 3, 1
    # pred_cluster_gt_id batch_size, 1, 1 

    gt_centorids_coords = gt_centorids_coords.permute(0,2,1)
    pred_centroid_coords = pred_centroid_coords.permute(0,2,1)

    nearby_centroids_dists = square_distance(pred_centroid_coords, gt_centorids_coords)
    _, nearby_centroids_indexes = nearby_centroids_dists.sort(dim=-1)
    nearby_centroids_indexes = nearby_centroids_indexes[:,:,0]
    pred_cluster_gt_ids = []
    for batch_idx in range(nearby_centroids_indexes.shape[0]):
        pred_cluster_gt_ids.append(gt_centroids_ids[batch_idx,0,nearby_centroids_indexes[batch_idx]])
    pred_cluster_gt_ids = torch.stack(pred_cluster_gt_ids, dim=0)
    return pred_cluster_gt_ids.view(-1,1,1)

def get_cluster_points_bin_label(gt_point_labels_inside_cluster, pred_cluster_gt_ids):
    # gt_point_labels_inside_cluster batch_size, 1, sampling num(4096)
    # cluster_points_labels batch_size, 1, num of points inside cluster
    # pred_cluster_gt_id batch_size, 1, 1 
    gt_labels = gt_point_labels_inside_cluster.view(gt_point_labels_inside_cluster.shape[0], gt_point_labels_inside_cluster.shape[2])
    centroid_labels = pred_cluster_gt_ids.view(pred_cluster_gt_ids.shape[0], pred_cluster_gt_ids.shape[2])
    gt_bin_label = torch.zeros_like(gt_labels)
    gt_bin_label[gt_labels == centroid_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    gt_bin_label = gt_bin_label.view(gt_labels.shape[0], 1, gt_labels.shape[1])

    return gt_bin_label

def get_gt_labels_nearest_points(cropped_coord, centroids, gt_labels):
    # 기각,,,
    # cropped_coords: batch_size, 3, num of points inside cluster 
    # centroid : batch_size, 3, 1
    # gt_labels: batch_size, 1(몇번인지), num of points inside cluster
    
    # gt_bin_label: batch_size, 1, cropped
    print(cropped_coord.shape)
    print(centroids.shape)
    centroids = centroids.permute(0,2,1)
    cropped_coord = cropped_coord.permute(0,2,1)

    nearby_points_dists = square_distance(centroids, cropped_coord)
    _, sorted_nearby_points_indexes = nearby_points_dists.sort(dim=-1)
    sorted_nearby_points_indexes = sorted_nearby_points_indexes.view(sorted_nearby_points_indexes.shape[0],sorted_nearby_points_indexes.shape[2])
    gt_near_labels = []

    for batch_idx in range(sorted_nearby_points_indexes.shape[0]):
        flat_gt_labels = gt_labels[batch_idx,:,:].view(-1)
        label_indexes, counts = flat_gt_labels[sorted_nearby_points_indexes[batch_idx, :100]].unique(sorted=True, return_counts = True)
        
        max_index = torch.argmax(counts[1:])
        max_label = torch.index_select(label_indexes[1:], 0, max_index)
        gt_near_labels += [max_label]
    gt_near_labels = torch.stack(gt_near_labels, dim=0)
    gt_near_labels = gt_near_labels.view(-1,1)

    gt_label = gt_labels.view(gt_labels.shape[0],-1)
    gt_bin_label = torch.zeros_like(gt_label)
    gt_bin_label[gt_label == gt_near_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    return gt_bin_label

def get_moved_sample_point(pred_offset, sampled_coord):
    # pred_offset : B, 24, encoder_sampled_points
    # sample_coord : B, 3, encoder_sampled_points
    # moved_coord: B, 16, 256(sampled_points), 3
    B, _, S_N = sampled_coord.shape
    sampled_coord = sampled_coord.permute(0,2,1)
    sampled_coord = sampled_coord.view(B, 1, S_N, 3)
    pred_offset = pred_offset.view(B, 16, 3, S_N)
    pred_offset = pred_offset.permute(0,1,3,2)
    #B, 8, encdoer_sampeld_points, 3
    moved_coord = sampled_coord + pred_offset
    return moved_coord

def get_moved_points_feature(sample_features, sample_coords, offset_result):
    #만들다 fp씀 그냥
    #sample_features: B, features(maybe 512), 256(sampled points)
    #sample_coords: B, 3(coords), 256(sampled_points)
    #offset_result: B, 3(coords_offset), 256(sampled_points)
    
    #return:moved_sampeld_points_features: B,16,256,features
     
    moved_sampled_points = get_moved_sample_point(offset_result, sample_coords)
    #moved_sampled_points: B, 16, 256(sampled_points), 3
    for i in range(16):
        class_moved_sampled_points = moved_sampled_points[:,i,:,:]

        dists = square_distance(moved_sampled_points, sample_coords)
        #dists에는, moved sampled points에서 주변점까지의 거리가 찍혀야 함

        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        #interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        #class_moved_sampled_points: B,256,3
        
def get_mean_centroids(dist_result, moved_points, class_num):
    #dist_result: 1, class_num, sampled_points_num
    #moved_points: 1, class_num, sampled_points_num, 3(coords)
    #return: class_num, 3
    moved_points_cpu_mean_ls = []
    for i in range(class_num):
        pred_dist_cpu = dist_result[0,i,:]
        moved_points_cpu = moved_points[0,i:i+1,:,:]

        moved_points_cpu = moved_points_cpu.cpu().detach().numpy()[0,:,:]
        pred_dist_cpu = pred_dist_cpu.cpu().detach().numpy()
        moved_points_cpu = moved_points_cpu[pred_dist_cpu<0.05]
        moved_points_cpu_mean_ls.append(np.mean(moved_points_cpu, axis=0))
    return moved_points_cpu_mean_ls
    
def get_dist_thr_labels(dist_pred):
    #dist_pred: B, 1, 16000
    dist_label = np.ones_like(dist_pred)
    dist_label[dist_pred>0.2] = 0
    return dist_label
    #ret: B, 1, 16000 짜리 label(0,1) by threshold 

def get_foreground_labels(dist_pred, pd_mask):
    #pd_mask: B, 1, 16000
    fg_label = np.ones_like(pd_mask)
    pd_mask = gen_utils.sigmoid(pd_mask)
    fg_label[pd_mask<0.5 | dist_pred>0.2] = 0
    return fg_label
    #ret: B, 1, 16000짜리 label(0,1) by threshold

def get_cropped_points_ls(pd_2, dist_pred, offset_pred, points, seg_label, centroid_coords):
    output_offset_cpu = offset_pred.cpu().detach().numpy()[0,:].T
    output_xyz_cpu = points[:,:3,:].cpu().detach().numpy()[0,:].T
    cen_cpu = centroid_coords.cpu().detach().numpy()[0,:].T
    output_dist_cpu = dist_pred.cpu().detach().numpy()[0,:].T.reshape(-1)
    output_input_xyz_cpu = points[:,:3,:].cpu().detach().numpy()[0,:].T

    fg_labels = gen_utils.get_foreground_labels(gen_utils.torch_to_numpy(pd_2), gen_utils.torch_to_numpy(dist_pred))
    fg_labels = fg_labels[0,:,:].T

    moved_point = output_offset_cpu+output_xyz_cpu
    fg_moved_point_result = np.concatenate([moved_point, fg_labels], axis=1)
    fg_moved_point_result = gen_utils.remove_0_points(fg_moved_point_result)

    cluster_mean_ls, labels = dbscan_pc(
        fg_moved_point_result, 
        dist_pred, 
        ret_per_labels = True, 
        method="aggl", 
        num_of_cluster=cen_cpu.shape[0]
    )
    #gen_utils.print_3d(gen_utils.np_to_pcd(sampled_db_scan[0], color=[0,1,0]), gen_utils.np_to_pcd(cen_cpu))
    
    fg_points = np.concatenate([output_xyz_cpu, fg_labels], axis=1)
    fg_points = gen_utils.remove_0_points(fg_points)
    fg_points = np.concatenate([fg_points, labels.reshape(-1,1)], axis=1)
    fg_points[:,3] +=1
    #gen_utils.print_3d(gen_utils.np_to_pcd_with_label(fg_points))

    bg_points = np.concatenate([output_xyz_cpu, fg_labels], axis=1)
    bg_points = gen_utils.remove_0_points(bg_points, target=1)
    bg_points = np.concatenate([bg_points, np.zeros([bg_points.shape[0],1])], axis=1)

    full_labeled_points = np.concatenate([fg_points, bg_points], axis=0)
    gen_utils.print_3d(gen_utils.np_to_pcd_with_label(full_labeled_points))
    
    points = points.permute(0,2,1)
    seg_label = seg_label.permute(0,2,1)
    cropped_points_ls = []
    cropped_seg_ls = []
    cropped_index_ls = []
    for i in range(1,17):
        teeth_arr = full_labeled_points[full_labeled_points[:,3]==i, :3]
        if teeth_arr.shape[0] == 0:
            continue
        #gen_utils.print_3d(gen_utils.np_to_pcd(teeth_arr))

        cropped_idxes = gen_utils.crop_bbox(points[0,:,:3], gen_utils.get_range_thr(teeth_arr), 0.02, True)
        cropped_index_ls.append(cropped_idxes)

        cropped_points = points[0, cropped_idxes, :]
        cropped_seg = seg_label[0, cropped_idxes, :]

        label_indexes, counts = cropped_seg.unique(sorted=True, return_counts = True)
        max_index = torch.argmax(counts)
        center_label_num = label_indexes[max_index]

        center_cond = cropped_seg[:,0] == center_label_num
        cropped_seg[:, 0] = -1
        if center_label_num != -1:
            cropped_seg[center_cond, 0] = 1

        cropped_points[:,:3] -= torch.mean(cropped_points[:,:3], dim=0)[0]
        cropped_seg_result = np.concatenate([gen_utils.torch_to_numpy(cropped_points)[:,:3], gen_utils.torch_to_numpy(cropped_seg)], axis=1)
        #gen_utils.print_3d(gen_utils.np_to_pcd_with_label(cropped_seg_result))
        cropped_points_ls.append(cropped_points)
        cropped_seg_ls.append(cropped_seg)

    #cropped_seg_ls = torch.stack(cropped_seg_ls)
    #cropped_seg_ls = cropped_seg_ls.permute(0,2,1)
    #cropped_points_ls = torch.stack(cropped_points_ls)
    #cropped_points_ls = cropped_points_ls.permute(0,2,1)

    return cropped_points_ls, cropped_seg_ls, cropped_index_ls

def centering_object(points):
    """crop시에 센터로 몰아넣는

    Args:
        points (torch array: B, feature(3까지는 coords인), N(sample num)): input features with coords

    Returns:
        torch array, B*feature*N(sample num): output
    """
    points = points.permute(0,2,1)
    for point in points:
        point[:,:3] -= torch.mean(point[:, :3], dim=0)
    points = points.permute(0,2,1)
    return points

def seg_label_to_cent(gt_coords, gt_seg_label):
    """
    change seg_label to centroids 16
    temporary batchsize is fixed as 1
    Input
    gt_coords: B, 3, 16000(sampling num) - coords
    gt_seg_label: B, 1, 16000(sampling num) - seg_labels 

    OUTPUT
    gt_cent_coords: B, 3, 16
    gt_cent_exists : B, 1, 16
    """
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