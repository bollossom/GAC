CUDA_VISIBLE_DEVICES=GPU_IDs python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 train_amp.py -net resnet104-b 256 -lr 0.1
