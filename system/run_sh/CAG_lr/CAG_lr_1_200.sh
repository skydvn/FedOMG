"""Cagrad learning rate"""
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5

