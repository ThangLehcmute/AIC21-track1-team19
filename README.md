# AIC21-track1-team19
## Introduction
### In this repo, we include the submission to AICity Challenge 2021 Vehicle Counts by Class at Multiple Intersections.

Our implementation comprised of:


(1) we re-designed a detection-tracking-counting (DTC) for movement-specific vehicle counting problem
regard to both effectiveness and efficiency.

(2) We modified Deep SORT with the efficient features to improve
the multiple objects tracking performance. 

(3) We proposed
the cosine similarity-based and orbit-based nearest
neighbor analysis to improve the vehicle counting
performance.
## Detector
### Download Dataset
1. Download images-set and lables : [link](https://drive.google.com/file/d/1wJSEGW2aamyeXmqSJuWgr2FyKScuuwL-/view?usp=sharing)

2. Unzip data, then move all to 'AIC21-track1-team19/ScaledYOLOv4-yolov4-csp/data/'.

### install mish-cuda
`
cd mish-cuda-master
python setup.py build install
`
### For install environment:
`
pip install -r requirements.txt`
### weights
1. Download weights : [link](https://drive.google.com/file/d/1Xlyd82J5J5Ktn73tYfgArPBk95kgxYFh/view?usp=sharing)

2. Unzip weights, then move all to 'AIC21-track1-team19/ScaledYOLOv4-yolov4-csp/'.


### Trainning
#### Go to 'ScaledYOLOv4-yolov4-csp'
#### you can change batch size to fit your GPU RAM.
`
python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --name yolov4-csp-3-0.25 --hyp ./data/hyp.finetune.yaml --img-size 512 512 --weight yolov4-csp.weights --epoch 300
`
##### For resume training: assume the checkpoint is stored in runs/exp0_yolov4-csp-3-0.25/weights/.
`
python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --weights 'runs/exp0_yolov4-csp-3-0.25/weights/last.pt' --name yolov4-csp-3-0.25 --resume
`
#### If you want to use multiple GPUs for training
`
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3 --batch-size 64 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --weights yolov4-csp.weights --name yolov4-csp-3-0.25 --sync-bn
`
## Counting Tracking and Creating submission csv file
### Structure
Directory structure:

* Dataset_A ([link](https://drive.google.com/drive/folders/1Q6s3YL0KQ2nnFM1Es8RvEAQcdfYEs_zf?usp=sharing))
* ScaledYOLOv4-yolov4-csp
* mish-cuda-master
* source_code
* weights

1. Go to 'AIC21-track1-team19/weigths'
2. Download weights: [link](https://drive.google.com/file/d/18ZbLNb1DfjJ42WGwQGMwRAfAJgOqqKm4/view?usp=sharing)
3. Go to 'AIC21-track1-team19/source_code/'
4. Run code
` python run_0304.py `
