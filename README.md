# Quantized Class Incremental Learning

Code for our paper ["Hadamard Domain Training with Integers for Class Incremental Quantized Learning"](https://web3.arxiv.org/abs/2310.03675) accepted for [Third Conference on Lifelong Learning Agents](https://lifelong-ml.cc/).

## Run Commands

No quant baseline CIL:
```python
python3 main.py -model icarl -p benchmark -seed 42467 --dataset="cifar100" --init_cls=20 --incre=20  --model_type="resnet32" --quantMethod="noq"
```

HDQT with CIL CIFAR100:
```python
python3 main.py -model icarl -p benchmark -seed 42467 --dataset="cifar100" --init_cls=20 --incre=20  --model_type="resnet32" --quantMethod="ours" --quantBits=4 --quantAccBits=8 --quantFWDWgt="int" --quantFWDAct="int" --quantBWDAct="stoch" --quantBWDWgt="int" --quantBWDGrad1="stoch" --quantBWDGrad2="stoch" --quantBlockSize=32
```

HDQT with CIL HAR-DSADS:
```python
python3 main.py -model icarl -p benchmark -seed 42467 --dataset="dsads" --init_cls=2 --incre=2  --model_type="fcnet" --fc_hid_dim=405 --init_lr=0.01 --lr=0.01 --epochs=100 --init_epoch=100 --memory_size=200 --init_milestones=50 --milestones=50 --quantMethod="ours" --quantBits=4 --quantAccBits=8 --quantFWDWgt="int" --quantFWDAct="int" --quantBWDAct="stoch" --quantBWDWgt="int" --quantBWDGrad1="stoch" --quantBWDGrad2="stoch" --quantBlockSize=32
```

LuQ [1] with CIL CIFAR100
```python
python3 main.py -model icarl -p benchmark -seed 42467 --dataset="cifar100" --init_cls=20 --incre=20  --model_type="resnet32" --quantMethod="luq_og" --quantBits=4 --quantAccBits=8
```

Supported CIL methods: icarl, bic, der, lwf, memo, ours  
Supported data sets: cifar100, dsads, hapt, pamap

All packages necessary to run commands can be found in requirements.txt

## Citation
```
@article{schiemer2023hadamard,
  title={Hadamard Domain Training with Integers for Class Incremental Quantized Learning},
  author={Schiemer, Martin and Schaefer, Clemens JS and Vap, Jayden Parker and Horeni, Mark James and Wang, Yu Emma and Ye, Juan and Joshi, Siddharth},
  journal={arXiv preprint arXiv:2310.03675},
  year={2023}
}
```

## Sources
[1] LuQ https://openreview.net/forum?id=yTbNYYcopd

[2] FP134 https://openreview.net/forum?id=3HJOA-1hb0e

[2] https://github.com/zhoudw-zdw/CIL_Survey 
