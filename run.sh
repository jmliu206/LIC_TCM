CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d [path of dataset] \
    --cuda --N 128 --lambda 0.05 --epochs 50 --lr_epoch 45 48 \
    --save_path [path for checkpoint] --save \
    --checkpoint [path of the pretrained checkpoint]


CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d /home/liu/dataset/dataset_30w/ \
    --cuda --N 128 --lambda 0.05 --epochs 50 --lr_epoch 45 48 \
    --save_path /home/liu/tmp/ --save \