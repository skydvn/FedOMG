"""gamma"""
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -mstr resnet10
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -mstr resnet10
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.2 -mstr resnet10
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1 -mstr resnet10