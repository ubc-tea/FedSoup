# debugging method usability
# python main.py -data retina -m resnet -algo Local -gr 500 -did 0 -eg 20 -go resnet_local -nc 3 -lr 1e-3 | tee ../tmp/retina/local_debug_console.output &
# python main.py -data retina -m resnet -algo FedAvg -gr 800 -did 0 -eg 20 -go fedavg_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedavg_debug_console.output
# python main.py -data retina -m resnet -algo FedProx -gr 800 -did 0 -eg 20 -go fedprox_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedprox_debug_console.output &
# python main.py -data retina -m resnet -algo FedDyn -gr 800 -did 0 -eg 20 -go feddyn_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/feddyn_debug_console.output &
# python main.py -data retina -m resnet -algo MOON -gr 800 -did 0 -eg 20 -go moon_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/moon_debug_console.output &
# python main.py -data retina -m resnet -algo FedBN -gr 800 -did 0 -eg 20 -go fedbn_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbn_debug_console.output &
# python main.py -data retina -m resnet -algo FedFomo -gr 800 -did 0 -eg 20 -go fedfomo_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedfomo_debug_console.output &
# python main.py -data retina -m resnet -algo FedRep -gr 800 -did 0 -eg 20 -go fedrep_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedrep_debug_console.output &
# python main.py -data retina -m resnet -algo FedBABU -gr 800 -did 0 -eg 20 -go fedbabu_debug -nc 3 -lr 1e-3 | tee ../tmp/retina/fedbabu_debug_console.output &
python main.py -data retina -m resnet -algo FedSoup -gr 800 -did 0 -eg 20 -go resnet -nc 3 -lr 1e-3 -wa_alpha 0.5 | tee ../tmp/retina/fedsoup_debug_console.output
