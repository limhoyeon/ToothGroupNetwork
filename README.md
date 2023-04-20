# ToothGroupNetwork
- Team CGIP
- Ho Yeon Lim and Min Chang Kim 

# Notice
- Thank you for pressing the star!
- This repository contains the code for the tooth segmentation algorithm that won first place at [Miccai2022 3D Teeth Scan Segmentation and Labeling Challenge
](https://3dteethseg.grand-challenge.org/evaluation/final-test-3d-teeth-segmentation-and-labeling/leaderboard/)
- We plan to publish a paper in May 2023 and will release additional training code and other networks (such as Tsegnet) used in the experiments accordingly.
- We used the dataset shared in the challenge, and since it is not our own data, it is not possible to share it.
- If you have any problem with execution, please contact me via email(hoyeon351@cglab.snu.ac.kr) or slack anytime. I'll give you a quick reply.
- trained checkpoints cannot be shared because of data copyright issue, so you have to train your model with your data. If you want test your data with our trained model, please request access in [Miccai2022 3D Teeth Scan Segmentation and Labeling Challenge](https://3dteethseg.grand-challenge.org/evaluation/final-test-3d-teeth-segmentation-and-labeling/leaderboard/) again. You can test your data with our trained models by requesting access in grand challenge platform. I'm sorry for the inconvenience. I'll try to find a way to share checkpoints from data provider.



# Inference
- We offer two model
  - TestModel: one for Preliminary Test phase - algorithm name "0726_bdl_v6"
  - FinalModel: another for Final Test phase - algorithm name "final_v1"
- All of the checkpoint files for each model are in (https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing). Download ckpts.zip and unzip all of checkpoints in ckpts folder.(*ckpts cannot be shared now, because of data copyright issue, please check notice.)
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
- All axes must be aligned as shown in the figure below. Note that the Y-axis points towards the back direction.
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
