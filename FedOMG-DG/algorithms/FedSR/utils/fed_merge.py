import torch
from torch.nn.utils import vector_to_parameters
import numpy as np

def Dict_weight(dict_in, weight_in):
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in
    
def Dict_Add(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Cal_Weight_Dict(dataset_dict, site_list=None):
    if site_list is None:
        site_list = list(dataset_dict.keys())
    weight_dict = {}
    total_len = 0
    for site_name in site_list:
        total_len += len(dataset_dict[site_name]['test'])
    for site_name in site_list:
        site_len = len(dataset_dict[site_name]['test'])
        weight_dict[site_name] = site_len/total_len
    return weight_dict

def FedAvg(model_dict, weight_dict, global_model=None):
    new_model_dict = None
    for model_name in weight_dict.keys():
        model = model_dict[model_name]
        model_state_dict = model.state_dict()
        if new_model_dict is None:
            new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
        else:
            new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
    
    if global_model is None:
        return new_model_dict
    else:
        global_model.load_state_dict(new_model_dict)
        return new_model_dict

def FedUpdate(model_dict, global_model):
    global_model_parameters = global_model.state_dict()
    for site_name in model_dict.keys():
        model_dict[site_name].load_state_dict(global_model_parameters)
    return None

def FedOMG(model_dict, weight_dict, global_model, dataobj, global_lr, cagrad_c):
    all_domain_grads = []
    flatten_global_weights = torch.cat([param.view(-1) for param in global_model.parameters()])
    for domain_name in dataobj.train_domain_list:
        domain_grad_diff = [torch.flatten(grad_param*weight_dict[domain_name] - global_param) for grad_param, global_param in 
               zip(model_dict[domain_name].parameters(), global_model.parameters())] 
        domain_grad_vector = torch.cat(domain_grad_diff)
        all_domain_grads.append(domain_grad_vector)
    
    all_domain_grads_tensor = torch.stack(all_domain_grads)

    omg_grads = OMG(all_domain_grads_tensor, 3, cagrad_c)
    flatten_global_weights += omg_grads * global_lr

    vector_to_parameters(flatten_global_weights, global_model.parameters())
    return None


def OMG(grad_vec, num_tasks, cagrad_c):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec
        print(grads.size())
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        print(GG.size())
        print(Gg.size())
        print(gg.size())

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg + 1e-4).sqrt() * cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c ** 2)
        return g

def MomentumUpdate(model, teacher, alpha=0.99):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k,v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1-alpha)*model_dict[k]
    teacher.load_state_dict(teacher_dict)