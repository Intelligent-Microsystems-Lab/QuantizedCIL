echo '-----------------------'
echo '-----------------------CNN lwf'
for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/lwf_cifar100_$1_${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,60}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------CNN icarl'
for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/icarl_cifar100_$1_${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,60}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------NME icarl'
for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/icarl_cifar100_$1_${i}.txt | grep -a 'NME top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,60}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------CNN bic'
for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/bic_cifar100_$1_${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,60}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------NME bic'
for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/bic_cifar100_$1_${i}.txt | grep -a 'NME top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,60}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done