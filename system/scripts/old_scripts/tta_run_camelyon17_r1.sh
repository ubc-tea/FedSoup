#!/bin/sh
# debugging method usability
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 20 -go resnet_local -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/local_debug_r1_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 20 -go fedavg_tta_r1_tmp -nc 5 -lr 1e-3 -tta True --save_img True | tee ../tmp/tiny_camelyon17/fedavg_tta_r1_console_tmp.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedProx -gr 1000 -did 0 -eg 20 -go fedprox_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/fedprox_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedDyn -gr 1000 -did 0 -eg 20 -go feddyn_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/feddyn_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo MOON -gr 1000 -did 0 -eg 20 -go moon_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/moon_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedBN -gr 1000 -did 0 -eg 20 -go fedbn_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/fedbn_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedFomo -gr 1000 -did 0 -eg 20 -go fedfomo_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/fedfomo_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedRep -gr 1000 -did 0 -eg 20 -go fedrep_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/fedrep_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedBABU -gr 1000 -did 0 -eg 20 -go fedbabu_tta_r1 -nc 5 -lr 1e-3 -tta True | tee ../tmp/tiny_camelyon17/fedbabu_tta_r1_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 20 -go fedsoup_tta_r1 -nc 5 -lr 1e-3 -tta True -wa_alpha 0.75 | tee ../tmp/tiny_camelyon17/fedsoup_tta_r1_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
