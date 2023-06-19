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
- train_data_split_txt_path과 val_data_split_txt_path를 통해 train/validation을 나누는 txt파일을 입력받습니다. 제공된 txt 파일을 이용하셔도 되고, 직접 split을 나누기 위해 txt파일을 만드셔도 됩니다.
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

### TSegNet
- [TSegNet](https://enigma-li.github.io/projects/tsegNet/TSegNet.html)을 구현한 모델입니다.

### pointnet


# Inference
- We offer two model
  - TestModel: one for Preliminary Test phase - algorithm name "0726_bdl_v6"
  - FinalModel: another for Final Test phase - algorithm name "final_v1"
- All of the checkpoint files for each model are in (https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing). Download ckpts.zip and unzip all of checkpoints in ckpts folder.
- The processing speed is not fast because the code has not been optimized yet.
- The processing speed of FinalModel is slow due to the use of ensamble method to increase performance. It is not clear whether this ensamble method actually increases accuracy.
- How to inference with TestModel
```
python inference_mid.py --input_path /your/input/path --save_path /your/save/path
```
- how to inference with FinalModel
```
python inference_final.py --input_path /your/input/path --save_path /your/save/path
```
- You can also inference with process_final.py and process_mid.py. It is same as process.py which was used for submission on grand challange platform.   
- The data structure for input is the same as the data provided during challenge (obj).
- You need to adhere to the data name format(casename_upper.obj or casename_lower.obj).
```
--input_path
----00OMSZGW
------00OMSZGW_lower.obj
------00OMSZGW_upper.obj
----0EAKT1CU
------0EAKT1CU_lower.obj
------0EAKT1CU_upper.obj.
and so on..
```
- predict results are saved in save_path like this
```
--save_path
----00OMSZGW_lower.json
----00OMSZGW_upper.json
----0EAKT1CU_lower.json
----0EAKT1CU_upper.json
and so on...
```
- All axes must be aligned as shown in the figure below. Note that the Y-axis points towards the back direction(plz note that both lower jaw and upper jaw have same z direction!)
![image](https://user-images.githubusercontent.com/70117866/233266358-1f7139ff-3921-44d8-b5bf-1461645de2b3.png)

- results are same as challenge json format which have "jaw", "instances" and "labels" as keys.


# How To Train Model
- if you want to train our model, please contact me.
- The training code will be updated and released during May 2023.

# installation
- Installtion is tested on pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel(ubuntu, pytorch 1.7.1)
- It can be installed on other OS(window, etc..)

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
