import torch
import time
from flcore.clients.clientCAGrad import clientCAGrad
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from scipy.optimize import minimize
import copy


class FedCAGrad(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCAGrad)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.update_grads = None
        self.cagrad_c = 0.5
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=1)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"global_model parameters grad")
            for param in self.global_model.parameters():
                print(param.grad)
            # print("model")
            # print(self.global_model.conv1[0].weight)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # for param in self.global_model.conv1.parameters():
            #     print(param.data)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.receive_grads()

            self.optimizer.zero_grad()
            grad_dims = []
            for mm in self.global_model.shared_modules():
                for param in mm.parameters():
                    grad_dims.append(param.data.numel())
            grads = torch.Tensor(sum(grad_dims), self.num_clients)
            # print(self.grads)

            for index, model in enumerate(self.grads):
                grad2vec(model, grads, grad_dims, index)
                self.global_model.zero_grad_shared_modules()
            g = cagrad_test(grads, alpha=0.5, rescale=1)
            overwrite_grad(self.global_model, g, grad_dims)
            # print(g)
            # self.optimizer.step()
            for param in self.global_model.parameters():
                param.data += 0.8 * param.grad

            print("Model_update")
            for param in self.global_model.parameters():
                print(param.grad)

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            # self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCAGrad)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


def cagrad_test(grads, alpha=0.5, rescale=1):
    # grads: [number_parameters, number_client]
    GG = grads.t().mm(grads).cpu()
    """
    GG = [C, P] * [P, C]
    GG size [num_client, num_client]
    [2P 4]
    [PP 2P]
    """
    g0_norm = (GG.mean() + 1e-8).sqrt()
    x_start = np.ones(2) / 2
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()
    # print(f"c_value: {c}")

    def objfn(x):
        return (x.reshape(1, 2).dot(A).dot(b.reshape(2, 1)) + c * np.sqrt(
            x.reshape(1, 2).dot(A).dot(x.reshape(2, 1)) + 1e-8)).sum()

    # print(objfn(x_start))
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    # print(res)
    w_cpu = res.x
    # print(f"w_cpu: {w_cpu}")
    ww = torch.Tensor(w_cpu).to(grads.device)
    # print(f"ww_size: {ww.size()}")
    gw = (grads * ww.view(1, -1)).sum(1)
    # print(f"gw_size: {gw.size()}")
    gw_norm = gw.norm()
    # print(f"gw_norm: {gw_norm}")
    lmbda = grads.mean(1).mean() * c / (gw_norm + 1e-8)
    # print(f"grads.mean(1): {grads.mean(1)}")
    g = grads.mean(1) + lmbda * gw
    # print(f"g_size: {g.size()}")
    # print(f"g: {g}")
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad_cur = p.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2  # to match the sum loss
    # print(f"newgrad: {newgrad}")
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            # print(f"param grad: {param.grad}")
            cnt += 1


























