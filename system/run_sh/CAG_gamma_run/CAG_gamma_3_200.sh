# """gamma"""
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -mstr resnet10
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -mstr resnet10
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.2 -mstr resnet10
# python main.py -data Cifar10 -gr 200 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1 -mstr resnet10


python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 -did 1
python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 -did 1
python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 -did 1
python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 -did 1