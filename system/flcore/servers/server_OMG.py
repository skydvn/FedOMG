import copy
import time
from flcore.clients.client_OMG import client_OMG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from utils.model_utils import ParamDict

class FedOMG(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(client_OMG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.omg_learning_rate = args.omg_learning_rate
        self.omg_step_size = args.omg_step_size
        self.omg_c = args.omg_c
        self.omg_rounds = args.omg_rounds
        self.omg_momentum = args.omg_momentum
        self.omg_gamma = args.omg_gamma

        self.omg_meta_lr = args.omg_meta_lr
        self.grad_balance = args.grad_balance


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

            meta_weights = self.omg_high(
                meta_weights=self.global_model,
                inner_weights=self.uploaded_models,
                lr_meta= self.omg_meta_lr
            )
            self.global_model.load_state_dict(copy.deepcopy(meta_weights))

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


    def omg_high(self, meta_weights, inner_weights, lr_meta):
        """
        Input:
        - meta_weights: class X(nn.Module)
        - inner_weights: list[X(nn.Module), X(nn.Module), ..., X(nn.Module)]
        - lr_meta: scalar value

        Output:
        - meta_weights: class X(nn.Module)

        """
        all_domain_grads = []
        flatten_meta_weights = torch.cat([param.view(-1) for param in meta_weights.parameters()])
        for i_domain in range(int(self.num_clients*self.jr)):
            domain_grad_diffs = [torch.flatten(inner_param - meta_param) for inner_param, meta_param in
                                 zip(inner_weights[i_domain].parameters(), meta_weights.parameters())]
            domain_grad_vector = torch.cat(domain_grad_diffs)
            all_domain_grads.append(domain_grad_vector)

        """
        - Grads normalization.
        """
        if self.grad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            domain_grad_norms = [torch.norm(grad) for grad in all_domain_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(domain_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in domain_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_retain_grads = [grad * scale for grad, scale in zip(domain_grad_norms, scaling_factors)]

            # Step 4: Stack the balanced gradients into a tensor
            all_domains_grad_tensor = torch.stack(balanced_retain_grads).t()
        else:
            all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        # print(all_domains_grad_tensor)
        g = self.omg_low(all_domains_grad_tensor, self.num_clients)

        flatten_meta_weights += g * lr_meta

        vector_to_parameters(flatten_meta_weights, meta_weights.parameters())
        meta_weights = ParamDict(meta_weights.state_dict())

        return meta_weights

    def omg_low(self, grad_vec, num_tasks):

        grads = grad_vec.to(self.device)

        GG = grads.t().mm(grads)
        # to(device)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
        #         w = torch.zeros(num_tasks, 1, requires_grad=True).to(self.device)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=self.omg_learning_rate * 2, momentum=self.omg_momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.omg_learning_rate, momentum=self.omg_momentum)

        scheduler = StepLR(w_opt, step_size=self.omg_step_size, gamma=self.omg_gamma)

        c = (gg + 1e-4).sqrt() * self.omg_c

        w_best = None
        obj_best = np.inf
        for i in range(self.omg_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.omg_rounds:
                obj.backward(retain_graph=True)
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.omg_c ** 2)
        return g


    def overwrite_grad2(self, m, newgrad):
        newgrad = newgrad * self.num_clients
        for param in m.parameters():

            num_elements = param.numel()

            param_slice = newgrad[:num_elements]

            param.grad = param_slice.view(param.data.size())

            newgrad = newgrad[num_elements:]




