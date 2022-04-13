CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/mcl_train.py -b 256 --height 180 --width 256 -a resnet50 -d market1501 --iters 200 --eps 0.6 --data-dir /root/data/zq/data/vessel_reid --logs-dir /root/vsislab-2/zq/ship_reid/cluster_contrast/log1

