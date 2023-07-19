#!/bin/sh
# debugging method usability
# CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo Local -gr 500 -did 0 -eg 20 -go resnet_local -nc 3 -lr 1e-3 | tee ../tmp/retina/local_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedAvg -gr 1000 -did 0 -eg 20 -go fedavg_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedavg_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedProx -gr 1000 -did 0 -eg 20 -go fedprox_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedprox_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedDyn -gr 1000 -did 0 -eg 20 -go feddyn_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/feddyn_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo MOON -gr 1000 -did 0 -eg 20 -go moon_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/moon_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBN -gr 1000 -did 0 -eg 20 -go fedbn_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbn_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedFomo -gr 1000 -did 0 -eg 20 -go fedfomo_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedfomo_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -eg 20 -go fedrep_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedrep_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -eg 20 -go fedbabu_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbabu_debug_r2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedSoup -gr 1000 -did 0 -eg 20 -go fedsoup_debug_r2 -nc 3 -lr 1e-3 -wa_alpha 0.75 | tee ../tmp/retina/fedsoup_debug_r2_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
