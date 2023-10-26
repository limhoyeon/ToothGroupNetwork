# ToothGroupNetwork
- Team CGIP
- Ho Yeon Lim and Min Chang Kim 

# Notice
- Please press the star!
- This repository contains the code for the tooth segmentation algorithm that won first place at [Miccai2022 3D Teeth Scan Segmentation and Labeling Challenge.](https://3dteethseg.grand-challenge.org/evaluation/challenge/leaderboard/)
- Official Paper for Challenge: [3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge.](https://arxiv.org/abs/2305.18277)
- We used the dataset shared in [the challenge git repository.](https://github.com/abenhamadou/3DTeethSeg22_challenge)
- If you only want the inference code, or if you want to use the same checkpoints that we used in the challenge, you can jump to [challenge_branch](https://github.com/limhoyeon/ToothGroupNetwork/tree/challenge_branch) in this repository.
- Most of the code has been modified. It may be significantly different from the code you downloaded before June 20th. If you downloaded this repository, we recommend download it again.
- There may be a lot of errors in the initial code. If you encounter any errors, please post them on the issue board, and we will promptly address and fix them(Im sorry for the slow resposes to issues because of personal reasons)...
- Please post your questions to the issue board as it's hard to see them in emails.

# Dataset
- We used the dataset shared in [the challenge git repository](https://github.com/abenhamadou/3DTeethSeg22_challenge). For more information about the data, check out the link.
- Dataset consists of dental mesh obj files and corresponding ground truth json files.
- You can also download the challenge training split data in [google drive](https://drive.google.com/drive/u/1/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby)(our codes are based on this data).
  - After you download and unzip these zip files, merge `3D_scans_per_patient_obj_files.zip` and `3D_scans_per_patient_obj_files_b2.zip`. The parent path of these obj files is `data_obj_parent_directory`.
  - Apply the same to the ground truth json files(`ground-truth_labels_instances.zip` and `ground-truth_labels_instances_b2.zip`. The parent path of these json files is `data_json_parent_directory`).
  - The directory structure of your data should look like below..
    ```
    --data_obj_parent_directory
    ----00OMSZGW
    ------00OMSZGW_lower.obj
    ------00OMSZGW_upper.obj
    ----0EAKT1CU
    ------0EAKT1CU_lower.obj
    ------0EAKT1CU_upper.obj
    and so on..
    
    --data_json_parent_directory
    ----00OMSZGW
    ------00OMSZGW_lower.json
    ------00OMSZGW_upper.jsno
    ----0EAKT1CU
    ------0EAKT1CU_lower.json
    ------0EAKT1CU_upper.json
    and so on..
    ```
- If you have your dental mesh data, you can use it.
  - In such cases, you need to adhere to the data name format(casename_upper.obj or casename_lower.obj).
  - All axes must be aligned as shown in the figure below. Note that the Y-axis points towards the back direction(plz note that both lower jaw and upper jaw have the same z-direction!).
    
    <img src="https://user-images.githubusercontent.com/70117866/233266358-1f7139ff-3921-44d8-b5bf-1461645de2b3.png" width="600" height="400">
  
# Training
## Preprocessing
- For training, you have to execute the `preprocess_data.py` to save the farthest point sampled vertices of the mesh (.obj) files.
- Here is an example of how to execute `preprocess_data.py`.
  ```
  python preprocess.py
   --source_obj_data_path data_obj_parent_directory \
   --source_json_data_path data_json_parent_directory \
   --save_data_path path/for/preprocessed_data
  ```

## Training
- We offer six models(tsegnet | tgnet(ours) | pointnet | pointnetpp | dgcnn | pointtransformer).
- For experiment tracking, we use [wandb](https://wandb.ai/site). Please replace "entity" with your own wandb ID in the `get_default_config function` of `train_configs/train_config_maker.py`.
- Due to the memory constraints, all functions have been implemented based on batch size 1 (minimum 11GB GPU RAM required). If you want to change the batch size to a different value, you will need to modify most of the functions by yourself.

### 1. tgnet(Ours)
- The tgnet is our 3d tooth segmentation method. Please refer to the [challenge paper](https://arxiv.org/abs/2305.18277) for an explanation of the methodology.
- You should first train the Farthest Point Sampling model and then train the Boundary Aware Point Sampling model.
- First, train the Farthest Point Sampling model as follows.
  ```
  start_train.py \
   --model_name "tgnet_fps" \
   --config_path "train_configs/tgnet_fps.py" \
   --experiment_name "your_experiment_name" \
   --input_data_dir_path "path/for/preprocessed_data" \
   --train_data_split_txt_path "base_name_train_fold.txt" \
   --val_data_split_txt_path "base_name_val_fold.txt"
  ```
  - Input the preprocessed data directory path into the `--input_data_dir_path`.
  - You can provide the train/validation split text files through `--train_data_split_txt_path` and `--val_data_split_txt_path`. You can either use the provided text files from the above dataset drive link(`base_name_*_fold.txt`) or create your own text files for the split.
- To train the Boundary Aware Point Sampling model, please modify the following four configurations in `train_configs/tgnet_bdl.py`: `original_data_obj_path`, `original_data_json_path`, `bdl_cache_path`, and `load_ckpt_path`.
  ![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/f4fc118e-6051-46a9-9862-d52f3d4ba2b9)
- After modifying the configurations, train the Boundary Aware Point Sampling model as follows.
  ```
  start_train.py \
   --model_name "tgnet_bdl" \
   --config_path "train_configs/tgnet_bdl.py" \
   --experiment_name "your_experiment_name" \
   --input_data_dir_path "path/to/save/preprocessed_data" \
   --train_data_split_txt_path "base_name_train_fold.txt" \
   --val_data_split_txt_path "base_name_val_fold.txt"
  ```

### 2. tsegnet
- This is the implementation of model [TSegNet](https://enigma-li.github.io/projects/tsegNet/TSegNet.html). Please refer to the paper for detail.
- First, The centroid prediction module has to be trained first in tsegnet. To train the centroid prediction module, please modify the `run_tooth_seg_mentation_module` parameter to False in the `train_configs/tsegnet.py` file.
  ![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/c37eb2ac-b36d-4ca9-a014-b785fd556c35)
- And train the centroid prediction module by entering the following command.
  ```
  start_train.py \
   --model_name "tsegnet" \
   --config_path "train_configs/tsegnet.py" \
   --experiment_name "your_experiment_name" \
   --input_data_dir_path "path/to/save/preprocessed_data" \
   --train_data_split_txt_path "base_name_train_fold.txt" \
   --val_data_split_txt_path "base_name_val_fold.txt"
  ```
- Once the training of the centroid prediction module is completed, please update the `pretrained_centroid_model_path` in `train_configs/tsegnet.py` with the checkpoint path of the trained centroid prediction module. Also, set `run_tooth_segmentation_module` to True.
- And please train the tsegnet model by entering the following command.
  ```
  start_train.py \
   --model_name "tsegnet" \
   --config_path "train_configs/tsegnet.py" \
   --experiment_name "your_experiment_name" \
   --input_data_dir_path "path/to/save/preprocessed_data" \
   --train_data_split_txt_path "base_name_train_fold.txt" \
   --val_data_split_txt_path "base_name_val_fold.txt"
  ```


### 3. pointnet | pointnetpp | dgcnn | pointtransformer
- [pointnet](https://arxiv.org/abs/1612.00593) | [pointnet++](http://stanford.edu/~rqi/pointnet2/) | [dgcnn](https://liuziwei7.github.io/projects/DGCNN) | [pointtransformer](https://arxiv.org/abs/2012.09164)
- This model directly applies the point cloud segmentation method to tooth segmentation.
- train models by entering the following command.
  ```
  start_train.py \
   --model_name "pointnet" \
   --config_path "train_configs/pointnet.py" \
   --experiment_name "your_experiment_name" \
   --input_data_dir_path "path/to/save/preprocessed_data" \
   --train_data_split_txt_path "base_name_train_fold.txt" \
   --val_data_split_txt_path "base_name_val_fold.txt"
  ```


# Inference
- To test the performance of the model used in the challenge, please switch to the challenge_branch (refer to the notice at the top).
- We offer six models(tsegnet | tgnet(ours) | pointnet | pointnetpp | dgcnn | pointtransformer).
- All of the checkpoint files for each model are in (https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing). Download ckpts(new).zip and unzip all of the checkpoints.
- Inference with tsegnet / pointnet / pointnetpp / dgcnn / pointtransformer
  ```
  python start_inference.py \
   --input_dir_path obj/file/parent/path \
   --split_txt_path base_name_test_fold.txt \
   --save_path path/to/save/results \
   --model_name tgnet_fps \
   --checkpoint_path your/model/checkpoint/path
  ```
- Inference with tgnet(ours)
  ```
  python start_inference.py \
   --input_dir_path obj/file/parent/path \
   --split_txt_path base_name_test_fold.txt \
   --save_path path/to/save/results \
   --model_name tgnet_fps \
   --checkpoint_path your/tgnet_fps/checkpoint/path
   --checkpoint_path_bdl your/tgnet_bdl/checkpoint/path
  ```
  - Please input the parent path of the original mesh obj files instead of the preprocessed sampling points in `--input_data_dir_path` for training. The inference process will handle the farthest point sampling internally.
  - For `split_txt_path`, provide the test split fold's casenames in the same format as used during training.
- Predicted results are saved in save_path like below... It has the same format as the ground truth json file.
  ```
  --save_path
  ----00OMSZGW_lower.json
  ----00OMSZGW_upper.json
  ----0EAKT1CU_lower.json
  ----0EAKT1CU_upper.json
  and so on...
  ```
- the inference config in "inference_pipelines.infenrece_pipeline_maker.py" has to be the same as the model of the train config. If you change the train config, then you have to change the inference config.

# Test results
- The checkpoints we provided were trained for 60 epochs using the train-validation split provided in the dataset drive link(`base_name_train_fold.txt`, `base_name_val_fold.txt`). The results obtained using the test split(`base_name_test_fold.txt`) are as follows
  
  ![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/507b0a8d-e82b-4acb-849d-86388c0099d3)
  
  (IoU -> Intersection over Union(TSA in challenge) // CLS -> classification accuracy(TIR in challenge)).

- The results may look like this.

  ![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/2d771b98-435c-49a6-827c-a85ab5bed6e2)


# Evaulation & Visualization
- We provide the evaluation and visualization code.
- You can execute the following code to test on a pair of obj/gt json file:
  ```
  eval_visualize_results.py \
   --mesh_path path/to/obj_file \ 
   --gt_json_path path/to/gt_json_file \
   --pred_json_path path/to/predicted_json_file(a result of inference code)
  ```
- With just a few modifications to the provided code, you can write code to test all the results.


# Installation
- Installtion is tested on pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel(ubuntu, pytorch 1.7.1) docker image.
- It can be installed on other OS(window, etc..)
- There are some issues with RTX40XX graphic cards. plz report in issue board.
- if you have any issues while installing pointops library, please install another pointops library source in (https://github.com/POSTECH-CVLab/point-transformer).
  ```
  pip install wandb
  pip install --ignore-installed PyYAML
  pip install open3d
  pip install multimethod
  pip install termcolor
  pip install trimesh
  pip install easydict
  cd external_libs/pointops && python setup.py install
  ```

# Reference codes
- https://github.com/LiyaoTang/contrastBoundary.git
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/POSTECH-CVLab/point-transformer.git
- https://github.com/fxia22/pointnet.pytorch
- https://github.com/WangYueFt/dgcnn
