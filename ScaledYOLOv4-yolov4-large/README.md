
## Installation

# install mish-cuda, if you use different pytorch version, you could try https://github.com/thomasbrandon/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```
## Testing

python test.py --img 512 --conf 0.001 --batch 8 --device 0 --data data/data.yaml --weights yolov4_csp_lar_512_sync
#weights have saving in runs/exp#/weights/best_<name> 

```

## Training
python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg ./models/yolov4-csp.yaml --name yolov4_csp_lar_512_sync --hyp ./data/hyp.finetune.yaml --img-size 512 512 --weight ./weights/yolov4-csp.weights --epoch 300 --sync-bn

```
##detect 
python detect.py --weights /weights/best_yolov4.pt --img 512 --conf 0.4 --source /cam_1.mp4