# debugging method usability
python main.py -data retina -m resnet -algo FedFomo -gr 1000 -did 0 -go fedfomo_per_level_M2 -nc 3 -lr 1e-3 -M 2 | tee ../tmp/retina/fedfomo_per_level_M2_console.output &
python main.py -data retina -m resnet -algo FedFomo -gr 1000 -did 0 -go fedfomo_per_level_M1 -nc 3 -lr 1e-3 -M 1 | tee ../tmp/retina/fedfomo_per_level_M1_console.output &
# python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -go fedrep_per_level_pls2 -nc 3 -lr 1e-3 -pls 2 | tee ../tmp/retina/fedrep_per_level_pls2_console.output &
# python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -go fedrep_per_level_pls3 -nc 3 -lr 1e-3 -pls 3 | tee ../tmp/retina/fedrep_per_level_pls3_console.output &
# python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -go fedrep_per_level_pls4 -nc 3 -lr 1e-3 -pls 4 | tee ../tmp/retina/fedrep_per_level_pls4_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedRep -gr 1000 -did 0 -go fedrep_per_level_pls5 -nc 3 -lr 1e-3 -pls 5 | tee ../tmp/retina/fedrep_per_level_pls5_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -go fedbabu_per_level_fts2 -nc 3 -lr 1e-3 -fts 2 | tee ../tmp/retina/fedbabu_per_level_fts2_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -go fedbabu_per_level_fts3 -nc 3 -lr 1e-3 -fts 3 | tee ../tmp/retina/fedbabu_per_level_fts3_console.output & 
# CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -go fedbabu_per_level_fts4 -nc 3 -lr 1e-3 -fts 4 | tee ../tmp/retina/fedbabu_per_level_fts4_console.output &
# CUDA_VISIBLE_DEVICES=1 python main.py -data retina -m resnet -algo FedBABU -gr 1000 -did 0 -go fedbabu_per_level_fts5 -nc 3 -lr 1e-3 -fts 5 | tee ../tmp/retina/fedbabu_per_level_fts5_console.output &

echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
