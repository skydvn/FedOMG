import copy
import time

from multiprocessing import pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *
from torch.nn.utils import vector_to_parameters

import wandb


class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam['num_rounds']
        self.fraction = hparam['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None
    
    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
    
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)
    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    

    def update_clients(self, sampled_client_indices):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(self._round)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        for idx in tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()
            
    

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)
    

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
                # print("DEBUG: server.py:183")
                # break
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            print(metric)
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_federated_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader)
                metric_dict[name] = metric
                
                if name == 'val':
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val
            
            print(metric_dict)
            if self.hparam['wandb']:
                wandb.log(metric_dict, step=self._round*self.hparam['local_epochs'])
            self.save_model(r)
            self._round += 1
        if self.hparam['wandb']:
            if best_id_val_round != 0: 
                wandb.summary['best_id_round'] = best_id_val_round
                wandb.summary['best_id_val_acc'] = best_id_val_test_value
            if best_lodo_val_round != 0:
                wandb.summary['best_lodo_round'] = best_lodo_val_round
                wandb.summary['best_lodo_val_acc'] = best_lodo_val_test_value
        else:
            print("best_id_round: " + best_id_val_round)
            print("best_id_val_acc: " + best_id_val_test_value)
            print("best_lodo_round: " + best_lodo_val_round)
            print("best_lodo_val_acc: " + best_lodo_val_test_value)
        self.transmit_model()

    def save_model(self, num_epoch):
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['id']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

class FedOMG(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.meta_lr = hparam['hparam3']
        self.cagrad_c = hparam['hparam4']
        self.caglr = hparam['hparam5']

    def omg(self, grad_vec, num_tasks, cagrad_c, caglr):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 100:
            w_opt = torch.optim.SGD([w], lr=caglr, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)
            # w_opt = torch.optim.Adam([w], lr=25)
        c = (gg + 1e-4).sqrt() * cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            # obj = ww.t().mm(Gg) + cagrad_c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            # obj = - ww.t().mm(Gg) - c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt() 
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                # obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        # lmbda = cagrad_c / (gw_norm + 1e-4)

        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c ** 2)

        return g

    def fed_omg(self, sampled_client_indices, coefficients, cagrad_c, meta_lr, caglr):
        all_domain_grads = []
        self.model.to("cpu")
        flatten_global_weights = torch.cat([param.view(-1) for param in self.model.parameters()])
        # averaged_weights = OrderedDict()

        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            domain_grad_diff = [torch.flatten(grad_param - global_param) * coefficients[it] for grad_param, global_param in
               zip(self.clients[idx].model.parameters(), self.model.parameters())]
            
            domain_grad_vector = torch.cat(domain_grad_diff)
            all_domain_grads.append(domain_grad_vector)

        all_domain_grads_tensor = torch.stack(all_domain_grads)
    
        omg_grads = self.omg(all_domain_grads_tensor, num_tasks=len(sampled_client_indices), cagrad_c=cagrad_c, caglr=caglr)
        # omg_grads.to("cpu")
        flatten_global_weights += omg_grads * meta_lr
    
        vector_to_parameters(flatten_global_weights, self.model.parameters())
        param_dict = ParamDict(self.model.state_dict())
        self.model.load_state_dict(param_dict)

        # self.model.load_state_dict(averaged_weights)
    def train_omg_model(self):
        """Do federated OMG training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.fed_omg(sampled_client_indices, mixing_coefficients, self.cagrad_c, self.meta_lr, self.caglr)

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_omg_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader)
                metric_dict[name] = metric
                
                if name == 'val':
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val
            
            print(metric_dict)
            if self.hparam['wandb']:
                wandb.log(metric_dict, step=self._round*self.hparam['local_epochs'])
            self.save_model(r)
            self._round += 1
        if self.hparam['wandb']:
            if best_id_val_round != 0: 
                wandb.summary['best_id_round'] = best_id_val_round
                wandb.summary['best_id_val_acc'] = best_id_val_test_value
            if best_lodo_val_round != 0:
                wandb.summary['best_lodo_round'] = best_lodo_val_round
                wandb.summary['best_lodo_val_acc'] = best_lodo_val_test_value
        else:
            print("best_id_round: " + best_id_val_round)
            print("best_id_val_acc: " + best_id_val_test_value)
            print("best_lodo_round: " + best_lodo_val_round)
            print("best_lodo_val_acc: " + best_lodo_val_test_value)
        self.transmit_model()

    def save_model(self, num_epoch):
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['id']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
    
class FedDG(FedAvg):
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
            client.set_amploader(self.amploader)
        super().register_clients(clients)
            
    def set_amploader(self, amp_dataset):
        self.amploader = amp_dataset


class FedADGServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.gen_input_size = int(hparam['hparam5'])

    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._generator = GeneDistrNet(num_labels=self.ds_bundle.n_classes, input_size=self.gen_input_size, hidden_size=self._featurizer.n_outputs)
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.generator = nn.DataParallel(self._generator)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier), copy.deepcopy(self._generator))

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict(), self._generator.state_dict())

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_model(self.model.state_dict(), self._generator.state_dict())
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            del message

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message

        averaged_weights = OrderedDict()
        averaged_generator_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            local_generator_weights = self.clients[idx].generator.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]                 
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]         
            for key in self.generator.state_dict().keys():
                if it == 0:
                    averaged_generator_weights[key] = coefficients[it] * local_generator_weights[key]
                    
                else:
                    averaged_generator_weights[key] += coefficients[it] * local_generator_weights[key]
        self.model.load_state_dict(averaged_weights)
        self.generator.load_state_dict(averaged_generator_weights)


class FedGMA(FedAvg):
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to('cpu')
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to('cpu')
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()
                # if it == 0:
                #     averaged_weights[key] = coefficients[it] * local_weights[key]
                # else:
                #     averaged_weights[key] += coefficients[it] * local_weights[key]
        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()
        # print(sign_delta[key])
        mask = abs_sign_delta.ge(self.hparam['hparam1'])
        # print("--mid--")
        # print(mask)
        # print("-------")
        final_mask = mask + (0-mask) * abs_sign_delta
        averaged_weights = last_weights + self.hparam['hparam1'] * final_mask * sum_delta 
        self.model.load_state_dict(averaged_weights)



class ScaffoldServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.c = None

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
                client.c_global = copy.deepcopy(self.c)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].c_global = copy.deepcopy(self.c)
    
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if it == 0:
                c_local = self.clients[idx].c_local
            else:
                c_local += self.clients[idx].c_local
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
    
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.c = c_local / len(sampled_client_indices)
        self.model.load_state_dict(averaged_weights)


class AFLServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_set = self.ds_bundle.dataset.get_subset('train', transform=self.ds_bundle.train_transform)
        train_g = self.ds_bundle.grouper.metadata_to_group(train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.is_group_in_train = is_group_in_train
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
       

    def transmit_lambda(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
            
                client.update_vector(self.group_weights)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_vector(self.group_weights)

    def aggregate(self,sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def update_lambda(self, sampled_client_indices):
        self.transmit_model(sampled_client_indices)
        total_loss_per_domain = torch.zeros_like(self.group_weights)
        total_samples_per_domain = torch.zeros_like(self.group_weights)
        # for client in tqdm(self.clients, leave=False):
        # # for client in self.clients:
        #     loss_per_domain, samples_per_domain = client.gradient_lambda()
        #     total_loss_per_domain += loss_per_domain
        #     total_samples_per_domain += samples_per_domain

        # send the global model to selected clients
        for idx in tqdm(sampled_client_indices, leave=False):
        # for idx in sampled_client_indices:
            loss_per_domain, samples_per_domain = self.clients[idx].gradient_lambda()
            total_loss_per_domain += loss_per_domain
            total_samples_per_domain += samples_per_domain
        self.group_weights += torch.nan_to_num(self.hparam['hparam1'] * total_loss_per_domain / total_samples_per_domain, nan=0.0)

        print(self.group_weights)

        self.group_weights = euclidean_proj_simplex(self.group_weights)

                
        print("after proj")
        print(self.group_weights)
        # print(self.group_weights)
        wandb.log({"l0_lmda": torch.count_nonzero(self.group_weights[self.group_weights>0.001])} ,step=self._round*self.hparam['local_epochs'])

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        self.transmit_lambda(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        self.update_lambda(sampled_client_indices)
