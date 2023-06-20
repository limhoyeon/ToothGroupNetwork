import sys
import os
from trimesh import PointCloud
sys.path.append(os.getcwd())
from glob import glob
import gen_utils as gu
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--mesh_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj", type=str)
parser.add_argument('--gt_json_path', default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances/013FHA7K/013FHA7K_lower.json" ,type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
args = parser.parse_args()


def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        #instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels==ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 += 2*(precision*recall) / (precision + recall)
        IOU += TP / (FP+TP+FN)
        IOU_arr.append(TP / (FP+TP+FN))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1
        #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr

gt_loaded_json = gu.load_json(args.gt_json_path)
gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)

pred_loaded_json = gu.load_json(args.pred_json_path)
pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)
_, mesh = gu.read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)
gu.print_3d(gu.get_colored_mesh(mesh, gt_labels)) # color is random
gu.print_3d(gu.get_colored_mesh(mesh, pred_labels)) # color is random