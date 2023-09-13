echo '-----------------------'
echo '-----------------------CNN lwf'
for i in 42467 88801 91280 01056 27534 81619 79004 25824 66362 33280 33150 27368 53375 70171 59431 14534 34018 85665 77797 17944 ; do cat logs/lwf_$1${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------CNN icarl'
for i in 42467 88801 91280 01056 27534 81619 79004 25824 66362 33280 33150 27368 53375 70171 59431 14534 34018 85665 77797 17944 ; do cat logs/icarl_$1${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------NME icarl'
for i in 42467 88801 91280 01056 27534 81619 79004 25824 66362 33280 33150 27368 53375 70171 59431 14534 34018 85665 77797 17944  ; do cat logs/icarl_$1${i}.txt | grep -a 'NME top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------CNN bic'
for i in 42467 88801 91280 01056 27534 81619 79004 25824 66362 33280 33150 27368 53375 70171 59431 14534 34018 85665 77797 17944  ; do cat logs/bic_$1${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done
echo '-----------------------NME bic'
for i in 42467 88801 91280 01056 27534 81619 79004 25824 66362 33280 33150 27368 53375 70171 59431 14534 34018 85665 77797 17944  ; do cat logs/bic_$1${i}.txt | grep -a 'NME top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done


# echo '-----------------------base'
# for i in 649323830 341384131 980310836 749032139 251745660 ; do cat logs/base_$1_${i}.txt | grep -a 'CNN top1 curve' | tail -1 | grep -o -P 'curve: \[.{0,80}' | sed 's/curve//' | sed 's/\: \[//' | sed 's/\]//'; done