# """gamma"""
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.2
python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1


python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 --noniid --balance --alpha_dirich 1 
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 1 
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1 -lbs 32 --noniid --balance --alpha_dirich 1