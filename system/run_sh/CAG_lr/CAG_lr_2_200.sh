# """Cagrad learning rate"""
python main.py -log -data Cifar10 -gr 800 -algo FedTest -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
python main.py -log -data Cifar10 -gr 800 -algo FedTest -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
python main.py -log -data Cifar10 -gr 800 -algo FedTest -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
python main.py -log -data Cifar10 -gr 800 -algo FedTest -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1

# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet8
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet8

# python main.py -data Cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 