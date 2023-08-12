# Quantized Class Incremental Learning

## sources
- https://openreview.net/forum?id=yTbNYYcopd 
- https://github.com/zhoudw-zdw/CIL_Survey 

```python
python3 main.py 
```

```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt run_icarl.script ${i} ${model} ${j}; done; done

for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt run_icarl.script ${i} ${model} ${j}; done; done

for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_sawb_bwd_g_ours_w_int_a_none_${i}.txt run_icarl.script ${i} ${model} ${j}; done; done

```
