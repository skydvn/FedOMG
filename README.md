# Federated Domain Generalization with Data-free On-server Gradient Matching - ICLR 2025 - Pytorch Official Implementation

# Abstract
<p align="justify">Domain Generalization (DG) aims to learn from multiple known source domains a model that can generalize well to unknown target domains. 
One of the key approaches in DG is training an encoder which generates domain-invariant representations. However, this approach is not applicable in Federated Domain Generalization (FDG), where data from various domains are distributed across different clients. In this paper, we introduce a novel approach, dubbed Federated Learning via On-server Matching Gradient (FedOMG), which can \emph{efficiently leverage domain information from distributed domains}. Specifically, we utilize the local gradients as information about the distributed models to find an invariant gradient direction across all domains through gradient inner product maximization. The advantages are two-fold: 1) FedOMG can aggregate the characteristics of distributed models on the centralized server without incurring any additional communication cost, and 2) FedOMG is orthogonal to many existing FL/FDG methods, allowing for additional performance improvements by being seamlessly integrated with them. Extensive experimental evaluations on various settings to demonstrate the robustness of FedOMG compared to other FL/FDG baselines. Our method outperforms recent SOTA baselines on four FL benchmark datasets (MNIST, EMNIST, CIFAR-10, and CIFAR-100), and three FDG benchmark datasets (PACS, VLCS, and OfficeHome).</p>

# Experiment
## Setup
This work can be conducted on any platform: Windows, Ubuntu, Google Colab. In Windows or Ubuntu use the following script to create a virtual environment.
```
git clone https://github.com/skydvn/FedOMG.git
cd path/to/FedOMG
python -m venv .env
```

## Running
For instance, the experiment of FedOMG by the following expressions:
```
python main.py -log -data Cifar10 -gr 800 -algo FedOMG -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 15 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1 
python main.py -log -data Cifar10 -gr 800 -algo FedOMG -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 50 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data Cifar10 -gr 800 -algo FedOMG -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 75 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1
python main.py -log -data Cifar10 -gr 800 -algo FedOMG -m resnet8 -mstr resnet8 -nc 100 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.5 -lbs 32 --noniid --balance --alpha_dirich 0.1
```

# Citation
```
@inproceedings{
  2025-FDG-FedOMG,
  title={Federated Domain Generalization with Data-free On-server Gradient Matching},
  author={Trong-Binh Nguyen and Minh-Duong Nguyen and Jinsun Park and Quoc-Viet Pham and Won Joo Hwang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  month = {May},
  year={2025},
  url={https://openreview.net/forum?id=8TERgu1Lb2}
}
```
