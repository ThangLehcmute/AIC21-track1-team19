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
1. Download images-set and lables : [link](https://drive.google.com/drive/folders/1x1bOvHdKnekuH2_-WPkXwaefIAbZFs6y?usp=sharing)

2. Go to 'ScaledYOLOv4-yolov4-large/data'

3. Create folders: 'train', 'train/images' and 'train/labels' into 'data' folder.

4. Unzip images-set, then move all images from images-set folder to 'ScaledYOLOv4-yolov4-large/data/train/images'.

5. Unzip labels, then move all json files to 'ScaledYOLOv4-yolov4-large/data/train/labels'.
### install mish-cuda
`
cd mish-cuda-master
python setup.py build install
`
### For install environment:
`
pip install -r requirements.txt`
### Pretrained Model
1. Download pretrained model : [link](https://drive.google.com/drive/folders/1FIIbxfv7gmS4bl33jlQiBOc8tC6LJBjM?usp=sharing)

2. Go to 'ScaledYOLOv4-yolov4-large/weights'

3. Put pretrained model in it

### Trainning
#### Go to 'ScaledYOLOv4-yolov4-large'
` python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg ./models/yolov4-csp.yaml --name yolov4_csp_lar_512_sync --hyp ./data/hyp.finetune.yaml --img-size 512 512 --weight ./weights/yolov4-csp.weights --epoch 300 --sync-bn
`
## Counting Tracking and Creating submission csv file
### Structure
Directory structure:

* Dataset_A ([link](https://drive.google.com/drive/folders/1Q6s3YL0KQ2nnFM1Es8RvEAQcdfYEs_zf?usp=sharing))
* ScaledYOLOv4-yolov4-large
* mish-cuda-master
* source_code
* weights

1. Go to 'AIC21-track1-team19/weigths'
2. Download weights: [link](https://drive.google.com/file/d/15Z-ZC101rXjL1TymFk8s4bPx2c28GXGx/view?usp=sharing)
3. Go to 'AIC21-track1-team19/source_code/'
4. Run code
` python run_0304.py `
