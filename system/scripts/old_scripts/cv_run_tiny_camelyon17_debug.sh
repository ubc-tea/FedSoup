#!/bin/sh
# debugging method usability
# python main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/local_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_hoid0 -nc 4 -lr 1e-3 -hoid 0 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_hoid0_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_hoid1 -nc 4 -lr 1e-3 -hoid 1 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_hoid1_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_hoid2 -nc 4 -lr 1e-3 -hoid 2 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_hoid2_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_hoid3 -nc 4 -lr 1e-3 -hoid 3 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_hoid3_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_hoid4 -nc 4 -lr 1e-3 -hoid 4 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_hoid4_debug_console.output &

# python main.py -data tiny_camelyon17 -m resnet -algo FedProx -gr 1000 -did 0 -eg 100 -go fedprox_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedprox_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo FedDyn -gr 1000 -did 0 -eg 100 -go feddyn_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/feddyn_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo MOON -gr 1000 -did 0 -eg 100 -go moon_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/moon_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo FedBN -gr 1000 -did 0 -eg 100 -go fedbn_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedbn_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo FedFomo -gr 1000 -did 0 -eg 100 -go fedfomo_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedfomo_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo FedRep -gr 1000 -did 0 -eg 100 -go fedrep_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedrep_debug_console.output &
# python main.py -data tiny_camelyon17 -m resnet -algo FedBABU -gr 1000 -did 0 -eg 100 -go fedbabu_debug -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedbabu_debug_console.output &

CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_new_hoid0 -nc 4 -lr 1e-3 -wa_alpha 0.75 -hoid 0 | tee ../tmp/tiny_camelyon17/fedsoup_new_hoid0_debug_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_new_hoid1 -nc 4 -lr 1e-3 -wa_alpha 0.75 -hoid 1 | tee ../tmp/tiny_camelyon17/fedsoup_new_hoid1_debug_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_new_hoid2 -nc 4 -lr 1e-3 -wa_alpha 0.75 -hoid 2 | tee ../tmp/tiny_camelyon17/fedsoup_new_hoid2_debug_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_new_hoid3 -nc 4 -lr 1e-3 -wa_alpha 0.75 -hoid 3 | tee ../tmp/tiny_camelyon17/fedsoup_new_hoid3_debug_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_new_hoid4 -nc 4 -lr 1e-3 -wa_alpha 0.75 -hoid 4 | tee ../tmp/tiny_camelyon17/fedsoup_new_hoid4_debug_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
