python main.py -log -data cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.7 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.2 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data cifar10 -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.1 -lbs 32 --noniid --balance --alpha_dirich 0.1


#python main.py -data mnist -gr 800 -algo FedTest -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 --noniid --balance --alpha_dirich 0.1