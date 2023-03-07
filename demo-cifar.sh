echo ID: CIFAR-10
echo With LINe Sparsity p_w=90 p_a=90 clip_threshold=1.0
python ood_eval.py --in-dataset CIFAR-10 --p-w 90 --p-a 90 --method taylor --clip_threshold 1.0

echo ID: CIFAR-100
echo With LINe Sparsity p_w=90 p_a=10 clip_threshold=1.0
python ood_eval.py --in-dataset CIFAR-100 --p-w 90 --p-a 10 --method taylor --clip_threshold 1.0
