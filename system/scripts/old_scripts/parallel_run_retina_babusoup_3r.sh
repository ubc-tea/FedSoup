#!/bin/sh
# debugging method usability
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 20 -go fedbabusoup_debug_r0 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbabusoup_debug_r0_console.output &
CUDA_VISIBLE_DEVICES=0 python main.py -data retina -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 20 -go fedbabusoup_debug_r1 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbabusoup_debug_r1_console.output &
CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABUSoup -gr 1000 -did 0 -eg 20 -go fedbabusoup_debug_r2 -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbabusoup_debug_r2_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
