# ToothGroupNetwork
- Team CGIP
- Ho Yeon Lim and Min Chang Kim 

# Notice
- Writing is in progress(This document will be translated into english...). It should be done soon and we will commit the training code as well.
- Please press the star!
- This repository contains the code for the tooth segmentation algorithm that won first place at [Miccai2022 3D Teeth Scan Segmentation and Labeling Challenge
](https://3dteethseg.grand-challenge.org/evaluation/challenge/leaderboard/)
- Official Paper for Challenge: [3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge](https://arxiv.org/abs/2305.18277)
- We used the dataset shared in [the challenge git repository](https://github.com/abenhamadou/3DTeethSeg22_challenge)
- If you only want the inference code, or if you want to use the same checkpoints that we used in the challenge, you can jump to [challenge_branch](https://github.com/limhoyeon/ToothGroupNetwork/tree/challenge_branch) in this repository.
- If you have any problem with execution, please contact me via email(hoyeon351@cglab.snu.ac.kr). I'll give you a quick reply.

# Data
- We used the dataset shared in [the challenge git repository](https://github.com/abenhamadou/3DTeethSeg22_challenge). For more information about the data, check out the link.
- You can also download the challenge training split data in [google drive](https://drive.google.com/drive/u/1/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby)(our codes are based on this data).
- Data consists of dental mesh obj files and corresponding ground truth json files.
- You need to adhere to the data name format(casename_upper.obj or casename_lower.obj).
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
- All axes must be aligned as shown in the figure below. Note that the Y-axis points towards the back direction(plz note that both lower jaw and upper jaw have the same z-direction!)
![image](https://user-images.githubusercontent.com/70117866/233266358-1f7139ff-3921-44d8-b5bf-1461645de2b3.png)

# Training
## Preprocessing
- 먼저, 빠른 학습을 위해 preprocess_data.py를 실행하여 mesh(.obj)의 vertex들을 farthest sampling한 결과를 저장합니다.
- 실행방법 예시는 아래와 같습니다.
```
python preprocess.py
 --source_obj_data_path data_obj_parent_directory \
 --source_json_data_path data_json_parent_directory \
 --save_data_path path/to/save/preprocessed_data
```

## Start train
- 총 6개의 모델을 제공합니다. (tsegnet | tgnet(ours) | pointnet | pointnetpp | dgcnn | pointtransformer)
- Experiment tracking을 위해 wandb를 사용합니다. train_configs/train_config_maker.py의 get_default_config 함수에서 "entity"를 본인의 wandb로 바꿔주세요.
- 메모리 문제로 batchsize 1을 기준으로 모든 함수가 구현되어 있습니다(GPU 최소 11GB ram 필요). batchsize를 다른 값으로 변경할 경우, 대부분의 함수를 직접 수정하셔야 합니다. 

### 1. tgnet(Ours)
- 챌린지에 참여한 모델입니다. 대략적인 네트워크 구조는 [논문](https://arxiv.org/abs/2305.18277) 확인해주세요.
- Farthest point sampling 모델을 먼저 학습한 뒤, Boundary Aware Point Sampling모델을 학습해야 합니다.
- 먼저, 아래와 같이 Farthest point sampling 모델을 학습해주세요.
```
start_train.py \
 --model_name "tgnet_fps" \
 --config_path "train_configs/tgnet_fps.py" \
 --experiment_name "your_experiment_name" \
 --input_data_dir_path "path/to/save/preprocessed_data" \
 --train_data_split_txt_path "base_name_train_fold.txt" \
 --val_data_split_txt_path "base_name_val_fold.txt"
```
- input_data_dir_path에는 앞서 preprocessing한 데이터를 입력하시면 됩니다.
- train_data_split_txt_path과 val_data_split_txt_path를 통해 train/validation을 나누는 txt파일을 입력받습니다. 데이터셋 드라이브 링크에 제공된 txt 파일을 이용하셔도 되고, 직접 split을 나누기 위해 txt파일을 만드셔도 됩니다.
- 그리고, Boundary Aware Point Sampling 모델을 학습하기 위해, train_configs/tgnet_bdl.py에서 아래와 같이 4개의 config를 수정해주세요(original_data_obj_path, original_data_json_path, bdl_cache_path, load_ckpt_path).
![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/f4fc118e-6051-46a9-9862-d52f3d4ba2b9)
- config를 수정한 뒤에, 아래와 같이 boundary aware point sampling 모델을 학습해주세요.
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
- [TSegNet](https://enigma-li.github.io/projects/tsegNet/TSegNet.html)을 구현한 모델입니다. 자세한 구조는 논문 참고해주세요.
- tsegnet은 먼저 centroid prediction module을 학습해야 합니다. 먼저, train_configs/tsegnet.py에서 run_tooth_seg_mentation_module 파라미터를 False로 바꿔주세요.

![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/c37eb2ac-b36d-4ca9-a014-b785fd556c35)
- 그리고, 아래와 같은 명령어를 입력하여 centroid prediction module을 학습해주세요.
```
start_train.py \
 --model_name "tsegnet" \
 --config_path "train_configs/tsegnet.py" \
 --experiment_name "your_experiment_name" \
 --input_data_dir_path "path/to/save/preprocessed_data" \
 --train_data_split_txt_path "base_name_train_fold.txt" \
 --val_data_split_txt_path "base_name_val_fold.txt"
```
- centroid prediction module 학습이 완료되면, train_configs/tsegnet.py에서 pretrained_centroid_model_path를 학습한 centroid prediction module의 checkpoint path로 바꿔주시고, run_tooth_segmentation_module을 True로 바꿔주세요!
- 그리고, 아래와 같은 명령어를 입력하여 tsegnet 모델을 학습해주세요.
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
- point cloud segmentation method를 바로 tooth segmentation에 적용한 모델입니다.
- 아래와 같은 명령어로 학습을 시작할 수 있습니다.
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
- 챌린지에 사용된 모델의 성능을 테스트하시려면, challenge_branch로 이동해주세요(상단 notice 참조).
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
- input_dir_path은 preprocessing한 sampling points들이 아닌, original mesh의 parent path를 입력해주세요(inference 내에서 farthest point sampling을 함).
- split_txt_path에는 train때와 같은 형식이지만, test split fold의 casename들을 넣어주세요.
- 결과 파일은 ground truth json file과 동일한 형식으로 생성됩니다. predict results are saved in save_path like this...
```
--save_path
----00OMSZGW_lower.json
----00OMSZGW_upper.json
----0EAKT1CU_lower.json
----0EAKT1CU_upper.json
and so on...
```
- the inference config in "inference_pipelines.infenrece_pipeline_maker.py" has to be the same as the model of the train config. If you changed the train config, then you have to change the inference config.  
# Test results
- 링크에 제공한 train, val split으로 총 60에포크 학습하고 test split으로 확인한 결과는 아래와 같습니다.
- IoU -> Intersection over Union(TSA in challenge) // CLS -> classification accuracy(TIR in challenge) 
![image](https://github.com/limhoyeon/ToothGroupNetwork/assets/70117866/507b0a8d-e82b-4acb-849d-86388c0099d3)

# Evaulation & Visualization
- Evaulation & Visualization을 제공합니다.
- 아래와 같이 실행하여, 한 개의 데이터에 대한 테스트를 실행할 수 있습니다.
```
eval_visualize_results.py \
 --mesh_path path/to/obj_file \ 
 --gt_json_path path/to/gt_json_file \
 --pred_json_path path/to/predicted_json_file(a result of inference code)
```
- 제공한 코드를 조금만 수정하면, 모든 결과에 대하여 테스트 할 수 있는 코드를 작성할 수 있습니다.


# installation
- Installtion is tested on pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel(ubuntu, pytorch 1.7.1) docker image.
- It can be installed on other OS(window, etc..)
- There are some issues with RTX40XX graphic cards. plz report in issue board.
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
