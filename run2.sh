# Define the list of variables
list=("42467"	"88801"	"91280"	"01056"	"27534"	"81619"	"79004"	"25824"	"66362"	"33280"	"33150"	"27368"	"53375"	"70171"	"59431"	"14534"	"34018"	"85665"	"77797"	"17944")

model="icarl"
bits="4"
accbits=("8" )
# quantMethod="luq_corrected"
quantMethod=("luq_corrected")

# Loop over the list
for abits in "${accbits[@]}"
do
for technique in "${quantMethod[@]}"
do
for i in "${list[@]}"
do
    bash run_dsads_acc.script "$i" "$model" "$bits" "$abits" "$technique" >> ./logs/${model}_cifar_${technique}_${bits}fwd_w_int_a_int_bwd_w_int_a_stoch_g1_stoch_g2_stoch_acc_${abits}_seed_${i}.txt
done
done
done