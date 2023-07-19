#!/bin/sh
# debugging method usability
# CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo Local -gr 500 -did 0 -eg 20 -go resnet_local -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/local_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedAvg -gr 1000 -did 0 -eg 20 -go fedavg_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedavg_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedProx -gr 1000 -did 0 -eg 20 -go fedprox_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedprox_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedDyn -gr 1000 -did 0 -eg 20 -go feddyn_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/feddyn_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo MOON -gr 1000 -did 0 -eg 20 -go moon_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/moon_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBN -gr 1000 -did 0 -eg 20 -go fedbn_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedbn_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedFomo -gr 1000 -did 0 -eg 20 -go fedfomo_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedfomo_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -eg 20 -go fedrep_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedrep_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -eg 20 -go fedbabu_tta_r2 -nc 5 -lr 1e-3 -tta True | tee ../tmp/retina/fedbabu_tta_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedSoup -gr 1000 -did 0 -eg 20 -go fedsoup_tta_r2 -nc 5 -lr 1e-3 -tta True -wa_alpha 0.75 | tee ../tmp/retina/fedsoup_tta_r2_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
