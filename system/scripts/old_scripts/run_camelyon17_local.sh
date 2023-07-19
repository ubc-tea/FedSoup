# debugging method usability
python main.py -data camelyon17 -m resnet -algo Local -gr 500 -did 0 -eg 10 -go fedavg_debug -nc 5 | tee ../tmp/cam_local_debug_console.output
