
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

python test.py --img 512 --conf 0.001 --batch 8 --device 0 --data data/data.yaml --weights yolov4-csp-3-0.25
#weights have saving in runs/exp#/weights/best_<name> 

```

## Training
# you can change batch size to fit your GPU RAM.
`
python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --name yolov4-csp-3-0.25 --hyp ./data/hyp.finetune.yaml --img-size 512 512 --weight yolov4-csp.weights --epoch 300
`
# For resume training: assume the checkpoint is stored in runs/exp0_yolov4-csp-3-0.25/weights/.
`
python train.py --device 0 --batch-size 8 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --weights 'runs/exp0_yolov4-csp-3-0.25/weights/last.pt' --name yolov4-csp-3-0.25 --
resume
`
# If you want to use multiple GPUs for training
`
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3 --batch-size 64 --data ./data/data.yaml --cfg yolov4-csp-3-0.25.cfg --weights yolov4-csp.weights --name yolov4-csp-3-0.25 --sync-bn
`

```
##detect 
python detect.py --weights /weights/best_yolov4-csp-3-0.25.pt --img 512 --conf 0.4 --source /cam_1.mp4
