
#python3 main.py -log -data Cifar10 -gr 100 -algo FedAvg -m cnn -nc 20 --noniid --balance --alpha_dirich 0.1 -did 1
#python3 main.py -log -data Cifar10 -gr 100 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
#python3 main.py -log -data Cifar10 -gr 100 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#
#python3 main.py -log -data mnist -gr 200 -algo FedCAG -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
#python3 main.py -log -data mnist -gr 400 -algo FedCAG -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#python3 main.py -log -data emnist -gr 200 -algo FedCAG -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#python3 main.py -log -data Cifar10 -gr 800 -algo FedROD -m resnet10 -mstr resnet10 -nc 20 --noniid --balance --alpha_dirich 0.1 -did 1

#python3 main.py -log -data mnist -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
#python main.py -log -data mnist -gr 400 -algo FedCAG -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1
#
#python main.py -log -data mnist -gr 100 -algo FedCagRod -m cnn -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1
#python main.py -log -data mnist -gr 200 -algo FedCagRod -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1
#python main.py -log -data mnist -gr 400 -algo FedCagRod -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1
#
#python main.py -data emnist -gr 100 -algo FedCAG -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1
#
#python3 main.py -log -data mnist -gr 100 -algo FedRod -m cnn -nc 20 --noniid --balance --alpha_dirich 0.1

#python3 main.py -log -data Cifar10 -gr 600 -algo FedCAG -m resnet10 -mstr resnet10 -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 --c_parameter 0.2
#python3 main.py -log -data Cifar10 -gr 600 -algo FedCAG -m resnet10 -mstr resnet10 -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 --c_parameter 2
#
#python3 main.py -log -data Cifar10 -gr 100 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 --c_parameter 5
#python3 main.py -log -data Cifar10 -gr 100 -algo FedCAG -m resnet10 -mstr resnet10 -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 --c_parameter 10

#python3 main.py -log -data mnist -gr 200 -algo FedCAG -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1
#python3 main.py -log -data mnist -gr 400 -algo FedCAG -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1

python3 main.py -log -data Cifar10 -gr 600 -algo FedCagRod -m resnet10 -mstr resnet10 -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 -jr 0.25
python3 main.py -log -data mnist -gr 600 -algo FedCagRod -m cnn -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 -jr 0.25

python3 main.py -log -data Cifar10 -gr 600 -algo FedCAG -m resnet10 -mstr resnet10 -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 -jr 0.25
python3 main.py -log -data mnist -gr 600 -algo FedCAG -m cnn -nc 80 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 -did 1 -jr 0.25