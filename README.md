# ToothGroupNetwork
- Team CGIP
- Ho Yeon Lim and Min Chang Kim 
- If you have any problem with execution, please contact me via email(hoyeon351@cglab.snu.ac.kr) or slack anytime. I'll give you a quick answer.

# Inference
- We offer two model
  - TestModel: one for Preliminary Test phase - algorithm name "0726_bdl_v6"
  - FinalModel: another for Final Test phase - algorithm name "final_v1"
- All of the checkpoint files for each model are in (https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing). Downloads ckpts.zip and unzip all of checkpoints in ckpts folder.
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
- you can also inference with process_final.py and process_mid.py. It is same as process.py which was used for submission on grand challange platform.   
- The data structure for input is the same as the data provided during challenge (obj).
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
- results are same as challenge json format which have "jaw", "instances" and "labels" as keys.


# How To Train Model
- if you want to train our model, please contact me

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
