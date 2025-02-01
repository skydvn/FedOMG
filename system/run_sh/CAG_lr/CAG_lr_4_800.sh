# """Cagrad learning rate"""
# python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5
# python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5
# python main.py -data Cifar10 -gr 800 -algo FedTest -nc 50 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5

python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 40 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1  
python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 40 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 
python main.py -log -data Cifar10 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1 
python main.py -log -data Cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 150 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1 -nb 100 