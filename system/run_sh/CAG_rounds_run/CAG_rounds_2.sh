# """Cagrad rounds"""
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 50 -calr 25 -mmt 0.5 -ss 15 -gam 0.5 -mstr resnet8
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet8
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -mstr resnet8
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 200 -calr 25 -mmt 0.5 -ss 40 -gam 0.5 -mstr resnet8

python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 200 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 3
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 50 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 3
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 3