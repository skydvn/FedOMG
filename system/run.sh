Default:
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
1: Cagrad rounds
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 50 -calr 25 -mmt 0.5 -ss 15 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 200 -calr 25 -mmt 0.5 -ss 40 -gam 0.5
2: Cagrad learning rate
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5
3: momentum
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.9 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.7 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.3 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.1 -ss 30 -gam 0.5
4: gamma
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.2
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1
5: resnet
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m resnet8
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m resnet10
6: local batch size
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m resnet8 -lbs 32
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m cnn -lbs 32
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m resnet10 -lbs 32
python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -m cnn -lbs 32

