#!/bin/sh
# debugging method usability
# python main.py -data tiny_camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 100 -go resnet_local -nc 4 -lr 1e-3 | tee ../tmp/tiny_camelyon17/local_debug_console.output &

CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 100 -go fedbabusoup_debug_hoid0 -nc 4 -lr 1e-3 -hoid 0 | tee ../tmp/tiny_camelyon17/fedbabusoup_debug_hoid0_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data tiny_camelyon17 -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 100 -go fedbabusoup_debug_hoid1 -nc 4 -lr 1e-3 -hoid 1 | tee ../tmp/tiny_camelyon17/fedbabusoup_debug_hoid1_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 100 -go fedbabusoup_debug_hoid2 -nc 4 -lr 1e-3 -hoid 2 | tee ../tmp/tiny_camelyon17/fedbabusoup_debug_hoid2_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 100 -go fedbabusoup_debug_hoid3 -nc 4 -lr 1e-3 -hoid 3 | tee ../tmp/tiny_camelyon17/fedbabusoup_debug_hoid3_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data tiny_camelyon17 -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 100 -go fedbabusoup_debug_hoid4 -nc 4 -lr 1e-3 -hoid 4 | tee ../tmp/tiny_camelyon17/fedbabusoup_debug_hoid4_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
