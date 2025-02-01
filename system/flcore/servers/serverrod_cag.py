import copy
import time
from flcore.clients.clientrod import clientROD
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import statistics

class FedCAG_ROD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientROD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.update_grads = None
        self.cagrad_c = args.c_parameter
        self.cagrad_rounds = args.cagrad_rounds
        self.cagrad_learning_rate = args.cagrad_learning_rate
        self.momentum = args.momentum
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.device = args.device
        model_origin = copy.deepcopy(args.model)

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

            grad_ez = sum(p.numel() for p in self.global_model.parameters())
            grads = torch.Tensor(grad_ez, self.num_clients)

            for index, model in enumerate(self.grads):
                grad2vec2(model, grads, index)

            g = self.cagrad(grads, self.num_clients)

            model_origin = copy.deepcopy(self.global_model)
            self.overwrite_grad2(self.global_model, g)
            for param in self.global_model.parameters():
                param.data += param.grad

            angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.grads]
            self.angle_value = statistics.mean(angle)
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientROD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def cagrad(self, grad_vec, num_tasks):

        grads = grad_vec.to(self.device)

        GG = grads.t().mm(grads)
        # to(device)
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=self.cagrad_learning_rate*2, momentum=self.momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.cagrad_learning_rate, momentum=self.momentum)

        scheduler = StepLR(w_opt, step_size=self.step_size, gamma=self.gamma)

        c = (gg+1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(self.cagrad_rounds+1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.cagrad_rounds:
                obj.backward()
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.cagrad_c**2)
        return g

    # def overwrite_grad(self, m, newgrad, grad_dims):
    #     newgrad = newgrad * self.num_clients  # to match the sum loss
    #     cnt = 0
    #     for mm in m.shared_modules():
    #         for param in mm.parameters():
    #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
    #             en = sum(grad_dims[:cnt + 1])
    #             this_grad = newgrad[beg: en].contiguous().view(param.data.size())
    #             param.grad = this_grad.data.clone()
    #             cnt += 1

    def overwrite_grad2(self, m, newgrad):
        newgrad = newgrad * self.num_clients
        for param in m.parameters():
            # Get the number of elements in the current parameter
            num_elements = param.numel()

            # Extract a slice of new_params with the same number of elements
            param_slice = newgrad[:num_elements]

            # Reshape the slice to match the shape of the current parameter
            param.grad = param_slice.view(param.data.size())

            # Move to the next slice in new_params
            newgrad = newgrad[num_elements:]


# def grad2vec(m, grads, grad_dims, task):
#     grads[:, task].fill_(0.0)
#     cnt = 0
#     for mm in m.shared_modules():
#         for p in mm.parameters():
#             grad_cur = p.data.detach().clone()
#             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#             en = sum(grad_dims[:cnt + 1])
#             grads[beg:en, task].copy_(grad_cur.data.view(-1))
#             cnt += 1


def grad2vec2(m, grads, task):
    grads[:, task].fill_(0.0)
    all_params = torch.cat([param.detach().view(-1) for param in m.parameters()])
    # print(all_params.size())
    grads[:, task].copy_(all_params)

