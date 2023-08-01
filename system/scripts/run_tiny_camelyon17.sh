#!/bin/sh

# python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/local_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_debug_0 -nc 5 -lr 1e-3 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedProx -gr 1000 -did 0 -eg 100 -go fedprox_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedprox_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedDyn -gr 1000 -did 0 -eg 100 -go feddyn_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/feddyn_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo MOON -gr 1000 -did 0 -eg 100 -go moon_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/moon_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedBN -gr 1000 -did 0 -eg 100 -go fedbn_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedbn_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedFomo -gr 1000 -did 0 -eg 100 -go fedfomo_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedfomo_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedRep -gr 1000 -did 0 -eg 100 -go fedrep_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedrep_debug_console.output &
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedBABU -gr 1000 -did 0 -eg 100 -go fedbabu_debug -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/fedbabu_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_debug -nc 5 -lr 1e-3 -wa_alpha 0.75 | tee ../tmp/tiny_camelyon17/fedsoup_debug_console.output &

# =========================================================
# debugging pruning

# FedAvg Pruning
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_p9_debug_0 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.9 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_p9_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_p8_debug_0 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.8 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_p8_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_p5_debug_0 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.5 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_p5_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedAvg -gr 1000 -did 0 -eg 100 -go fedavg_p0_debug_0 -nc 5 -lr 1e-3 2>&1 | tee ../tmp/tiny_camelyon17/fedavg_p0_debug_console.output &

# FedSoup Pruning
CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_p9_debug -nc 5 -lr 1e-3 -wa_alpha 0.75 --pruning --sparsity_ratio 0.9 | tee ../tmp/tiny_camelyon17/fedsoup_p9_debug_console.output &

CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_p8_debug -nc 5 -lr 1e-3 -wa_alpha 0.75 --pruning --sparsity_ratio 0.8 | tee ../tmp/tiny_camelyon17/fedsoup_p8_debug_console.output &

CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_p5_debug -nc 5 -lr 1e-3 -wa_alpha 0.75 --pruning --sparsity_ratio 0.5 | tee ../tmp/tiny_camelyon17/fedsoup_p5_debug_console.output &

CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo FedSoup -gr 1000 -did 0 -eg 100 -go fedsoup_p0_debug -nc 5 -lr 1e-3 -wa_alpha 0.75 | tee ../tmp/tiny_camelyon17/fedsoup_p0_debug_console.output &


# Local Pruning
# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local_p9 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.9 | tee ../tmp/tiny_camelyon17/local_p9_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local_p8 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.8 | tee ../tmp/tiny_camelyon17/local_p8_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local_p5 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.5 | tee ../tmp/tiny_camelyon17/local_p5_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local_p1 -nc 5 -lr 1e-3 --pruning --sparsity_ratio 0.1 | tee ../tmp/tiny_camelyon17/local_p1_debug_console.output &

# CUDA_VISIBLE_DEVICES=0 python -u main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local_p0 -nc 5 -lr 1e-3 | tee ../tmp/tiny_camelyon17/local_p0_debug_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
