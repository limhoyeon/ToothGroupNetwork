import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import json
from external_libs.pointops.functions import pointops

#from testmodel.modules.pointnext.pt_models.layers.subsample import furthest_point_sample
import trimesh

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def np_to_pcd_with_prob(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])

    base_color = np.array([1,0,0])
    base_color_2 = np.array([0,1,0])
    label_colors = np.zeros((arr.shape[0], 3))
    for idx in range(label_colors.shape[0]):
        label_colors[idx] = arr[idx, 3] * (base_color) + (1-arr[idx, 3]) * base_color_2
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def save_pcd(path, arr):
    o3d.io.write_point_cloud(path, arr)

def save_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def count_unique_by_row(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def load_colored_mesh(org_mesh_path, ipr_mesh_path, stl_path_list, up_down_idx, matching_dist_thr):
    global_mesh = load_mesh(org_mesh_path[up_down_idx])
    global_mesh = global_mesh.remove_duplicated_vertices()
    global_mesh_arr = np.asarray(global_mesh.vertices)

    cluster_vertex_colors = np.ones((np.asarray(global_mesh.vertices).shape))

    tree = KDTree(global_mesh_arr, leaf_size=2)

    cluster_vertex_colors = np.zeros((np.asarray(global_mesh.vertices).shape))
    cluster_vertex_colors[:,0] = 1

    ipr_mesh = load_mesh(ipr_mesh_path[up_down_idx])
    ipr_mesh = ipr_mesh.remove_duplicated_vertices()
    ipr_mesh_arr = np.asarray(ipr_mesh.vertices)

    dists, indexs = tree.query(ipr_mesh_arr, k=10)

    for point_num, (corr_idx_ls, dist_ls) in enumerate(zip(indexs,dists)):
        not_matching_flag=True

        for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
            if dist_item<0.0001:
                cluster_vertex_colors[idx_item,:] = np.asarray([0,1,1])
                not_matching_flag = False
                
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(cluster_vertex_colors)
    total_ls = []
    for idx in range(len(stl_path_list)):
        stl_path = stl_path_list[idx]

        tooth_num = get_number_from_name(stl_path)
        if up_down_idx==1:
            if(tooth_num>=30):
                continue
        else:
            if(tooth_num<30):
                continue

        #tooth_num = get_number_from_name(stl_path)%10
        #if tooth_num == 9:
        #    continue
        #if get_number_from_name(stl_path)//10 in [2,4]:
        #    tooth_num += 8


        #gin_ls = read_txt_ls(gin_txt_path_list[idx])
        #total_ls.append(gen_utils.np_to_pcd(gin_ls))
        
        mesh = load_mesh(stl_path)
        mesh_arr = np.asarray(mesh.vertices)
        dists, indexs = tree.query(mesh_arr, k=4)
        tooth_num_color = np.random.rand(3)
        tooth_num_color[0] = tooth_num/50
        for point_num, (corr_idx_ls, dist_ls) in enumerate(zip(indexs,dists)):
            #True리면 해당 label은 아무 것도 색칠 못 했다는 것,,
            #하지만 무조건 하나는 색칠해야 한다.
            not_matching_flag=True
            for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
                if dist_item < 0.0001:
                    cluster_vertex_colors[idx_item,:] = np.asarray(tooth_num_color)
                    not_matching_flag = False
                    
            if not_matching_flag:
                for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
                    if dist_item< matching_dist_thr and (cluster_vertex_colors[idx_item,:] == np.array([1,0,0])).all():
                        cluster_vertex_colors[idx_item,:] = np.asarray(tooth_num_color)

    for i in range(cluster_vertex_colors.shape[0]):
        if (cluster_vertex_colors[i,:] == np.array([1,0,0])).all():
            cluster_vertex_colors[i,:] = np.array([0,1,1])
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(cluster_vertex_colors)
    #print_3d(global_mesh)
    return global_mesh


def recomp_normals(arr):
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(arr[:,3:])
    pcd.estimate_normals(search_param)
    return np.array(pcd.normals)

def load_mesh(mesh_path, only_tooth_crop = False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    if only_tooth_crop:
        cluster_idxes, cluster_nums, _ = mesh.cluster_connected_triangles()
        cluster_idxes = np.asarray(cluster_idxes)
        cluster_nums = np.asarray(cluster_nums)
        tooth_cluster_num = np.argmax(cluster_nums)
        mesh.remove_triangles_by_mask(cluster_idxes!=tooth_cluster_num)
    return mesh

def colorling_mesh_with_label(mesh, label_arr, colorling):
    label_arr = label_arr.reshape(-1)
    if colorling=="sem":
        palte = np.array([
            [255,153,153],

            [153,76,0],
            [153,153,0],
            [76,153,0],
            [0,153,153],
            [0,0,153],
            [153,0,153],
            [153,0,76],
            [64,64,64],

            [255,128,0],
            [153,153,0],
            [76,153,0],
            [0,153,153],
            [0,0,153],
            [153,0,153],
            [153,0,76],
            [64,64,64],
        ])/255
    else:
        palte = np.random.rand(200,3)
        palte[0,:] = np.array([255,153,153]) 
    palte[9:] *= 0.4

    verts_arr = np.array(mesh.vertices)
    label_colors = np.zeros((verts_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)




def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    if type(label_arr) == np.ndarray:
        arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:,axis] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def np_to_pcd_removed(arr):
    points = arr[:,:3]
    labels = arr[:,3]
    points = points[labels==1, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def remove_0_points(arr, target=0):
    #arr 16000, 4
    #ret 0이 아닌 거의 개수, 3
    points = arr[:,:3]
    labels = arr[:,3]
    points = points[labels==(1-target), :]
    return points

def get_gt_mask_labels(pred_mask_1, pred_weight_1, gt_label):
    # pred_mask_1: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped

    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_label)
    gt_bin_label[gt_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.long)
    
    return gt_bin_label

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def np_to_by_label(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2],
       [0.32736046, 0.71189952, 0.20750141],
       [0.56743345, 0.07504726, 0.34684285],
       [0.35949841, 0.4314623 , 0.98791015],
       [0.31151589, 0.44971993, 0.86484811],
       [0.96808667, 0.42096273, 0.95791817],
       [0.64136201, 0.41471365, 0.11085843],
       [0.4789342 , 0.30820783, 0.34425576],
       [0.50173988, 0.38319907, 0.09296238]])
    
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def resample_pcd(pcd_ls, n, method):
    """Drop or duplicate points so that pcd has exactly n points"""
    if method=="uniformly":
        idx = np.random.permutation(pcd_ls[0].shape[0])
    elif method == "fps":
        idx = new_fps(pcd_ls[0][:,:3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    """
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    """
    return pcd_resampled_ls


def fps_depr(xyz, npoint):
    #xyz : N,3
    xyz = torch.from_numpy(np.array([xyz])).type(torch.float).cuda()
    idx = furthest_point_sample(xyz, npoint)
    return torch_to_numpy(idx[0,:])

def new_fps(xyz, npoint):
    if xyz.shape[0]<=npoint:
        raise "new fps error"
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(xyz, torch.tensor([xyz.shape[0]]).cuda().type(torch.int), torch.tensor([npoint]).cuda().type(torch.int)) 
    return torch_to_numpy(idx).reshape(-1)

def fps(xyz, npoint):
    xyz = torch.from_numpy(xyz).cuda()
    xyz = xyz.view(1,-1,3).type(torch.float)
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return torch_to_numpy(centroids).reshape(-1)

def cropped_to_global_label_marker(org_points, cropped_tooth_exists, cropped_weights, cropped_tooth_num, nearest_indexes, binary_output=False):
    # org_points : 3, 16000
    # cropped_tooth_exists : num of cluster, 1 , num of points inside cluster
    # cropped_weights : num of cluster, 1 , num of points inside cluster
    # cropped_tooth_num : num of cluster, 8
    # nearest_indexes : num of cluster, num of points inside cluster

    labeled_points = org_points.cpu().detach().numpy().T
    cropped_tooth_num = cropped_tooth_num.cpu().detach().numpy()
    cropped_tooth_exists = cropped_tooth_exists.view(cropped_tooth_exists.shape[0], cropped_tooth_exists.shape[2])
    cropped_weights = cropped_weights.view(cropped_weights.shape[0], cropped_weights.shape[2])
    labeled_points = np.concatenate([labeled_points, np.zeros((labeled_points.shape[0],1))], axis=1)
    for cluster_idx in range(len(nearest_indexes)):
        labeled_points[nearest_indexes[cluster_idx]]

        cropped_tooth_exists_inside_cluster = cropped_tooth_exists[cluster_idx, :].cpu().detach().numpy()
        cropped_tooth_weights_inside_cluster = cropped_weights[cluster_idx, :].cpu().detach().numpy()
        cropped_tooth_exists_inside_cluster = sigmoid(cropped_tooth_exists_inside_cluster)
        cropped_tooth_weights_inside_cluster = sigmoid(cropped_tooth_weights_inside_cluster)
        cropped_indexes_have_label_arr = nearest_indexes[cluster_idx][(cropped_tooth_exists_inside_cluster)>=0.5]
        if binary_output:
            labeled_points[cropped_indexes_have_label_arr, 3] = 1
        else:
            labeled_points[cropped_indexes_have_label_arr, 3] = np.argmax(cropped_tooth_num[cluster_idx])+1
    return labeled_points

def print_3d(*data_3d_ls, point_show_normal = True):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_wireframe = True, mesh_show_back_face = True)

def print_capture_3d(data_3d_ls, save_path):
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.savefig(save_path)
        vis.close()
        return True
    o3d.visualization.draw_geometries_with_animation_callback(data_3d_ls,capture_image)

def get_bounding_box_line_set(mean_point, regr, color=[1,0,0]):
    #mean_point: [x,y,z]
    #regr: [w,h,depth, exist]
    #basis -> 3(3개의 축),3(normal vector)

    x_min = mean_point - np.array([regr[0]/2,0,0])
    x_max = mean_point + np.array([regr[0]/2,0,0])
    y_min = mean_point - np.array([0,regr[1]/2,0])
    y_max = mean_point + np.array([0,regr[1]/2,0])
    z_min = mean_point - np.array([0,0,regr[2]/2])
    z_max = mean_point + np.array([0,0,regr[2]/2])

    #points = [x_min,x_max,y_min,y_max,z_min,z_max]
    #points = np.array(points)
    y_interval = (y_max-y_min)/2
    z_interval = (z_max-z_min)/2
    #top_points = [x_min + z_interval, x_min + - z_interval, x_min + z_interval, x_min - z_interval ]
    top_points = [x_min + y_interval + z_interval, x_min + y_interval - z_interval, x_min -y_interval - z_interval, x_min - y_interval + z_interval ]
    bottom_points = [x_max + y_interval + z_interval, x_max + y_interval - z_interval, x_max -y_interval - z_interval,  x_max - y_interval + z_interval ]
    points = top_points + bottom_points
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color]*12)
    """
    points = [[x_min, y_min, z_min], [x_min,y_max,z_min], [x_max,y_min,z_min], [x_max, y_max,z_max],
             [x_min, y_min, z_max], [x_min,y_max,z_max], [x_max,y_min,z_max], [x_max, y_max,z_max]
             ]
    """
    return line_set


def wdh_to_6points(mean_point, regr):
    #mean_point: [x,y,z]
    #regr: [w,h,depth, exist]
    x_min = mean_point[0] - regr[0]/2
    x_max = mean_point[0] + regr[0]/2
    y_min = mean_point[1] - regr[1]/2
    y_max = mean_point[1] + regr[1]/2
    z_min = mean_point[2] - regr[2]/2
    z_max = mean_point[2] + regr[2]/2
    points = [x_min,x_max,y_min,y_max,z_min,z_max]
    return points

def cal_iou(points1, points2):
    overlap = (min(points1[1],points2[1]) - max(points1[0],points2[0])) * \
            (min(points1[3],points2[3]) - max(points1[2],points2[2])) * \
            (min(points1[5],points2[5]) - max(points1[4],points2[4]))
    points1_area = (points1[1]-points1[0]) * (points1[3]-points1[2]) * (points1[5]-points1[4])
    points2_area = (points2[1]-points2[0]) * (points2[3]-points2[2]) * (points2[5]-points2[4])
    
    return (overlap) / (points1_area + points2_area - overlap)
    
def NMS(bbox_regr, center_point, pred_exist):
    #bbox_regr: 256, 3 
    #center_point 256, 3
    #pred_exist: 256
    sorted_arg = np.argsort(pred_exist)
    sorted_arg = sorted_arg[::-1][:len(sorted_arg)]
    

    pred_exist = pred_exist[sorted_arg]
    center_point = center_point[sorted_arg]
    bbox_regr = bbox_regr[sorted_arg]
    points_arr = []
    for bbox_regr_idx in range(bbox_regr.shape[0]):
        points_arr.append(wdh_to_6points(center_point[bbox_regr_idx,:], bbox_regr[bbox_regr_idx,:]))
    points_arr = np.array(points_arr)

    remain_bbox_regr_idx = [0]
    for bbox_regr_idx in range(1, points_arr.shape[0]):
        if np.count_nonzero(bbox_regr[bbox_regr_idx,:]<0):
            continue
        to_be_removed_flag=False
        for target_remain_bbox_regr_idx in remain_bbox_regr_idx:
            if cal_iou(points_arr[target_remain_bbox_regr_idx], points_arr[bbox_regr_idx])>=0.2:
                to_be_removed_flag=True
        
        if not to_be_removed_flag:
            remain_bbox_regr_idx.append(bbox_regr_idx) 

    return bbox_regr[remain_bbox_regr_idx], center_point[remain_bbox_regr_idx], pred_exist[remain_bbox_regr_idx]

def get_dist_thr_labels(dist_pred):
    #dist_pred: B, 1, 16000
    dist_label = np.ones_like(dist_pred)
    dist_label[dist_pred>0.2] = 0
    return dist_label
    #ret: B, 1, 16000 짜리 label(0,1) by threshold 

def get_foreground_labels(pd_mask, dist_pred=None, pd_mask_thr = 0.5):
    #pd_mask: B, 1, 16000
    fg_label = np.ones_like(pd_mask)
    pd_mask = sigmoid(pd_mask)
    if dist_pred is None:
        cond = (pd_mask<pd_mask_thr)
    else:
        cond = (pd_mask<pd_mask_thr) | (dist_pred>0.2)
    fg_label[cond] = 0
    return fg_label
    #ret: B, 1, 16000짜리 label(0,1) by threshold

def get_weight_thr_labels(weight_pred):
    #dist_pred: B, 1, 16000
    dist_label = np.ones_like(weight_pred)
    weight_pred = sigmoid(weight_pred)
    dist_label[weight_pred>0.4] = 0
    return dist_label
    #ret: B, 1, 16000 짜리 label(0,1) by threshold 

def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()


Y_AXIS_SCALING = True

def save_json_infer_label_chl(mesh_path, points_with_label):
    Y_AXIS_MAX = 33.15232091532151
    Y_AXIS_MIN = -36.9843781139949
    global_mesh = o3d.io.read_triangle_mesh(mesh_path)
    global_mesh = global_mesh.remove_duplicated_vertices()

    vertices = np.asarray(global_mesh.vertices)
    vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
    vertices[:, :3] = ((vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX-Y_AXIS_MIN))*2-1

    tree = KDTree(points_with_label[:,:3], leaf_size=2)
    ind = tree.query(vertices, k=1, return_distance = False)
    vertex_labels = (points_with_label[ind.reshape(-1),3]).astype(int)

    #normal_colors = (np.random.rand(17,3)+0.3)/1.3
    #normal_colors[0,:3] = np.array([1,1,1])
    normal_colors = [[1,1,1],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2]]
    normal_colors += [[1/2,1/2,0], [1/2,0,1/2], [0,1/2,1/2], [1,1/2,0], [1,0,1/2], [0,1,1/2],[1/2,1,0], [1/3,1/3,1]]
    normal_colors = [[1,1,1],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2]]
    normal_colors += [[1/2,1/2,0], [1/2,0,1/2], [0,1/2,1/2], [1,1/2,0], [1,0,1/2], [0,1,1/2],[1/2,1,0], [1/3,1/3,1]]

    vertex_colors = np.ones(vertices.shape)
    for vertex_num in range(vertex_colors.shape[0]):
        vertex_colors[vertex_num, :] = normal_colors[vertex_labels[vertex_num]]
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return global_mesh
    
def load_mesh_with_label(mesh_path, points_with_label):
    Y_AXIS_MAX = 44.7410888671875
    Y_AXIS_MIN = -45.46401596069336
    global_mesh = o3d.io.read_triangle_mesh(mesh_path)
    global_mesh = global_mesh.remove_duplicated_vertices()

    is_up = get_up_from_name(mesh_path)
    if is_up:
        R = global_mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
        global_mesh.rotate(R, center=(0, 0, 0))

    global_mesh_arr = np.asarray(global_mesh.vertices)
    global_mesh_arr[:, :3] = ((global_mesh_arr[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX-Y_AXIS_MIN))*2-1

    #global_mesh_arr[:, :3] -= np.mean(points_with_label[:,:3], axis=0)
    tree = KDTree(points_with_label[:,:3], leaf_size=2)
    ind = tree.query(global_mesh_arr, k=1, return_distance = False)
    vertex_labels = (points_with_label[ind.reshape(-1),3]).astype(int)

    #normal_colors = (np.random.rand(17,3)+0.3)/1.3
    #normal_colors[0,:3] = np.array([1,1,1])
    normal_colors = [[1,1,1],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2]]
    normal_colors += [[1/2,1/2,0], [1/2,0,1/2], [0,1/2,1/2], [1,1/2,0], [1,0,1/2], [0,1,1/2],[1/2,1,0], [1/3,1/3,1]]

    vertex_colors = np.ones(global_mesh_arr.shape)
    for vertex_num in range(vertex_colors.shape[0]):
        vertex_colors[vertex_num, :] = normal_colors[vertex_labels[vertex_num]]
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return global_mesh

def get_range_thr(arr):
    if type(arr) is np.ndarray:
        max_thr = np.max(arr[:,:3], axis=0)
        min_thr = np.min(arr[:,:3], axis=0)
        return np.concatenate([min_thr, max_thr])
    else:
        max_thr = torch.max(arr[:,:3], dim=0)[0]
        min_thr = torch.min(arr[:,:3], dim=0)[0]
        return torch.cat([min_thr, max_thr])

def crop_bbox(arr, range_thr, margin, index = False):
    cond = (arr[:,0] >= (range_thr[0] - margin)) & \
        (arr[:,0] <= (range_thr[3] + margin)) & \
        (arr[:,1] >= (range_thr[1] - margin)) & \
        (arr[:,1] <= (range_thr[4] + margin)) & \
        (arr[:,2] >= (range_thr[2] - margin-0.02)) & \
        (arr[:,2] <= (range_thr[5] + margin+0.02))
    if index:
        return cond.nonzero().view(-1)
    else:
        return arr[cond, :]

def get_random_cluster_idxes(max_range, num_to_genetrate):
    rand_cluster_idx = []
    for i in range(num_to_genetrate):
        for count in range(10):
            rand_num = np.random.randint(0, max_range)
            if (rand_num not in rand_cluster_idx):
                rand_cluster_idx.append(rand_num)
                break
            else:
                rand_num = np.random.randint(0, 16)
    return rand_cluster_idx

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)

def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr


def axis_rotation(axis, angle):
    ang = np.radians(angle) 
    R=np.zeros((3,3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang)+ux*ux*(1-cos(ang))
    R[0][1] = ux*uy*(1-cos(ang)) - uz*sin(ang)
    R[0][2] = ux*uz*(1-cos(ang)) + uy*sin(ang)
    R[1][0] = uy*ux*(1-cos(ang)) + uz*sin(ang)
    R[1][1] = cos(ang) + uy*uy*(1-cos(ang))
    R[1][2] = uy*uz*(1-cos(ang))-ux*sin(ang)
    R[2][0] = uz*ux*(1-cos(ang))-uy*sin(ang)
    R[2][1] = uz*uy*(1-cos(ang))+ux*sin(ang)
    R[2][2] = cos(ang) + uz*uz*(1-cos(ang))
    return R

def make_coord_frame(size=1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])

def cum_avg (prevAvg, newNumber, listLength):
    oldWeight = (listLength - 1) / listLength
    newWeight = 1 / listLength
    return (prevAvg * oldWeight) + (newNumber * newWeight)

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def save_json(file_path, json_obj):
    with open(file_path, "w") as json_file:

        json.dump(json_obj, json_file)

def read_txt(file_path):
    f = open(file_path, 'r')
    path_ls = []
    while True:
        line = f.readline().split()
        if not line: break
        path_ls.append(os.path.join(os.path.dirname(file_path), line.split("\n")[0] + ".npy"))
    f.close()

    return path_ls

def draw_axis(vectors, mean_point, length=10, type="first", only_half=False):
    points = []
    lines = []
    if type=="first":
        colors = [[1,0,0], [0,1,0], [0,0,1]]
    else:
        colors = [[1,1,0], [0,1,1], [1,0,1]]

    for idx, vector in enumerate(vectors):
        if only_half:
            points.append(mean_point)
        else:
            points.append(mean_point-length*vector)
        points.append(mean_point+length*vector)
        lines.append([2*idx,2*idx+1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def read_txt_obj_ls(path, ret_mesh=False, load_color=False, use_tri_mesh=False, ret_area_ls = False):
    # tri mesh will change vertex order
    if use_tri_mesh:
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
        if ret_area_ls:
            area_ls = np.array(tri_mesh_loaded_mesh.area_faces)
        #vertex_color_ls =[[]]
    else:
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        #vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0]=='v':
                vertex_ls.append(list(map(float,line[1:4])))
                #vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0]=='f':
                tri_verts_idxes = list(map(str,line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    #if len(vertex_color_ls[0]) != 0:
    #    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_color_ls)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)


    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls,norms], axis=1)]

    if ret_mesh:
        output.append(mesh)
    if ret_area_ls:
        output.append(area_ls)
    return output