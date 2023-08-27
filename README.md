# Quantized Class Incremental Learning

## sources
- https://openreview.net/forum?id=yTbNYYcopd 
- https://github.com/zhoudw-zdw/CIL_Survey 

```python
python3 main.py 
```

```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt run_cifar.script ${i} ${model} ${j}; done; done

for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt run_cifar.script ${i} ${model} ${j}; done; done

for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i} -o ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt  -e ./logs/${model}_cifar100_${j}fwd_w_int_a_int_bwd_g1_stoch_g2_stoch_w_int_a_stoch_${i}.txt run_cifar.script ${i} ${model} ${j}; done; done


```

## HAR


### DSADS
```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_dsads.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_dsads.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_dsads_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_dsads.script ${i} ${model} ${j}; done; done
```


### HAPT

```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_hapt.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_hapt.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_hapt_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_hapt.script ${i} ${model} ${j}; done; done
```

### PAMAP - Problem so far

```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_pamap.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_pamap.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_pamap_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_pamap.script ${i} ${model} ${j}; done; done
```


### WISDM - Problem so far

```bash
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=bic; qsub  -N ${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_wisdm.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=icarl; qsub  -N ${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_wisdm.script ${i} ${model} ${j}; done; done
for i in 649323830 341384131 980310836 749032139 251745660 ; do for j in 4; do model=lwf; qsub  -N ${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i} -o ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt  -e ./logs/${model}_wisdm_${j}fwd_w_int_a_int_bwd_g1_int_g2_int_w_int_a_stoch_${i}.txt run_wisdm.script ${i} ${model} ${j}; done; done
```
