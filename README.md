# Learned Image Compression with Mixed Transformer-CNN Architectures [CVPR23 Highlight]

This is the repository of the paper "[Learned Image Compression with Mixed Transformer-CNN Architectures](https://arxiv.org/abs/2303.14978)"

## Abstract
Learned image compression (LIC) methods have exhibited promising progress and superior rate-distortion performance compared with classical image compression standards. Most existing LIC methods are Convolutional Neural Networks-based (CNN-based) or Transformer-based, which have different advantages. Exploiting both advantages is a point worth exploring, which has two challenges: 1) how to effectively fuse the two methods? 2) how to achieve higher performance with a suitable complexity? In this paper, we propose an efficient parallel Transformer-CNN Mixture (TCM) block with a controllable complexity to incorporate the local modeling ability of CNN and the non-local modeling ability of transformers to improve the overall architecture of image compression models. Besides, inspired by the recent progress of entropy estimation models and attention modules, we propose a channel-wise entropy model with parameter-efficient swin-transformer-based attention (SWAtten) modules by using channel squeezing. Experimental results demonstrate our proposed method achieves state-of-the-art rate-distortion performances on three different resolution datasets (i.e., Kodak, Tecnick, CLIC Professional Validation) compared to existing LIC methods.

## Architectures
The overall framework.

<img src="./assets/tcm.png"  style="zoom: 33%;" />

The proposed entropy model.

<img src="./assets/entropy.png"  style="zoom: 33%;" />

## Evaluation Results
RD curves on Kodak.

<img src="./assets/res.png"  style="zoom: 33%;" />

## Training
``` 
CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d [path of training dataset] \
    --cuda --N 128 --lambda 0.05 --epochs 50 --lr_epoch 45 48 \
    --save_path [path for checkpoint] --save \
    --checkpoint [path of the pretrained checkpoint]
```

## Testing
``` 
python eval.py --checkpoint [path of the pretrained checkpoint] --data [path of testing dataset] --cuda
```

## Pretrained Model
| Lambda | Metric | Link |
|--------|--------|------|
| 0.05   | MSE    |   [link](https://drive.google.com/file/d/1TK-CPiD2QwtWJqZoT_OyCtnxdQ7UNP56/view?usp=share_link)   |


## Reconstructed Samples
<img src="./assets/visual.png"  style="zoom: 33%;" />


## Citation
```
@inproceedings{liu2023tcm,
  author = {Liu, Jinming and Sun, Heming and Katto, Jiro},
  title = {Learned Image Compression with Mixed Transformer-CNN Architectures},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1--10},
  year = {2023}
}
```

