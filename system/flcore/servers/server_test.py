import copy
import time
from flcore.clients.client_test import client_test
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import torch
from scipy.optimize import minimize

class FedTest(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(client_test)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.update_grads = None
        self.cagrad_c = 0.5
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=10)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )
        self.server_grads = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.receive_grads()

            self.server_grads = copy.deepcopy(self.grads)
            for model in self.server_grads:
                for param in model:
                    param.data.zero_()

            # for agg_model, old_model in zip(self.server_grads, self.grads):
            #     for agg_param, up_param in zip(agg_model.parmeters(), old_model.parameters()):
            #         agg_param.data =
            #

            grad_dims = []
            for mm in self.global_model.shared_modules():
                for param in mm.parameters():
                    grad_dims.append(param.data.numel())
            grads = torch.Tensor(sum(grad_dims), self.num_clients)
            # print(grad_dims)
            # print(grads.size())
            # size(582026, 2)

            for index, model in enumerate(self.grads):
                grad2vec(model, grads, grad_dims, index)
                self.global_model.zero_grad_shared_modules()
            # g = cagrad_test(grads, alpha=0.5, rescale=1)
            g, ww = self.cagrad(grads, self.num_clients)

            self.overwrite_grad(self.global_model, g, grad_dims)
            for param in self.global_model.parameters():
                param.data += param.grad

            # for param in self.global_model.parameters():
            #     print(param.grad)

            # if self.dlg_eval and i % self.dlg_gap == 0:
            #     self.call_dlg(i)
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

    def cagrad(self, grad_vec, num_tasks):

        grads = grad_vec
        # size(582026, num_clients)

        GG = grads.t().mm(grads).cpu()
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        # gg is scalar

        w = torch.zeros(num_tasks, 1, requires_grad=True)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            # size (num_clients, 1)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.cagrad_c**2)
        return g

    def overwrite_grad(self, m, newgrad, grad_dims):
        newgrad = newgrad * self.num_clients  # to match the sum loss
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
