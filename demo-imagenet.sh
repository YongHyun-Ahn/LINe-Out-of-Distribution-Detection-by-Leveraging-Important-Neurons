
echo ID: ImageNet
echo LINe
python ood_eval.py --name resnet50 --in-dataset imagenet --p-w 10 --p-a 10 --method taylor --clip_threshold 0.8

