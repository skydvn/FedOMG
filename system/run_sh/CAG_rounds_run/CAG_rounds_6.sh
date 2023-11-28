
"""Cagrad rounds"""
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 50 -calr 25 -mmt 0.5 -ss 15 -gam 0.5 -mstr resnet10
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet10
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet10
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 200 -calr 25 -mmt 0.5 -ss 40 -gam 0.5 -mstr resnet10