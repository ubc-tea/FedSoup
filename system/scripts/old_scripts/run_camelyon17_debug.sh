# debugging method usability
python main.py -data camelyon17 -m resnet -algo FedAvg -gr 500 -did 0 -eg 10 -go fedavg_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedavg_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedProx -gr 500 -did 0 -eg 50 -go fedprox_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedprox_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedDyn -gr 500 -did 0 -eg 50 -go feddyn_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/feddyn_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo MOON -gr 500 -did 0 -eg 50 -go moon_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/moon_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedBN -gr 500 -did 0 -eg 50 -go fedbn_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedbn_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedFomo -gr 500 -did 0 -eg 50 -go fedfomo_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedfomo_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedRep -gr 500 -did 0 -eg 50 -go fedrep_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedrep_debug_cam_console.output
# python main.py -data camelyon17 -m resnet -algo FedBABU -gr 500 -did 0 -eg 50 -go fedbabu_debug_cam -nc 5 --batch_size 128 | tee ../tmp/camelyon17/fedbabu_debug_cam_console.output
