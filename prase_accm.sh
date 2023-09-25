echo '-----------------------'
echo '-----------------------CNN bic'
for j in 4 12 16; do echo ${j}; for i in 42467 88801 91280; do cat logs/bic_cifar_$1${j}_${i}.txt  | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done; done

echo '-----------------------CNN icarl'
for j in 12; do echo ${j}; for i in 42467 88801 91280; do cat logs/icarl_cifar_4fwd_w_int_a_int_bwd_w_int_a_stoch_g1_stoch_g2_stoch_acc${j}_${i}.txt  | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done; done

echo '-----------------------CNN lwf'
for j in 12; do echo ${j}; for i in 42467 88801 91280; do cat logs/lwf_cifar_4fwd_w_int_a_int_bwd_w_int_a_stoch_g1_stoch_g2_stoch_acc${j}_${i}.txt  | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done; done

# lwf_cifar_4fwd_w_int_a_int_bwd_w_int_a_stoch_g1_stoch_g2_stoch_acc4_42467.txt
# bic_cifar_4fwd_w_int_a_int_bwd_w_int_a_stoch_g1_stoch_g2_stoch_acc12_91280