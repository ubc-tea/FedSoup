# python main.py -data retina -m resnet -algo FedSoup -gr 800 -did 0 -eg 20 -go resnet -nc 3 -lr 1e-3 -wa_alpha 0.25 | tee ../tmp/retina/fedsoup0.25round_debug_console.output
python main.py -data retina -m resnet -algo FedSoup -gr 800 -did 0 -eg 20 -go resnet -nc 3 -lr 1e-3 -wa_alpha 0.5 | tee ../tmp/retina/fedsoup0.5round_debug_console.output
# python main.py -data retina -m resnet -algo FedSoup -gr 800 -did 0 -eg 20 -go resnet -nc 3 -lr 1e-3 -wa_alpha 0.75 | tee ../tmp/retina/fedsoup0.75round_debug_console.output
