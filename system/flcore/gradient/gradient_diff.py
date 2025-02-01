import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import gradient_utils
from collections import OrderedDict
from numbers import Number
import operator

class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def diff_weight(self, model1, model2):
        params1 = [p.data for p in model1.parameters()]
        params2 = [p.data for p in model2.parameters()]

        # Tính hiệu và norm của hiệu giữa các parameter tương ứng
        diff_norms = [torch.norm(p1 - p2, p='fro') for p1, p2 in zip(params1, params2)]

        # Tính tổng (hoặc trung bình) của các norm này để có một đại lượng đơn lẻ mô tả sự khác biệt
        total_diff_norm = torch.sum(torch.stack(diff_norms))
        return total_diff_norm.item()

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        # print(f"prev:{prev_param[0]}")
        # print(f"p1:{params1[0]}")
        # print(f"p2:{params2[0]}")

        grad1 = params1 - prev_param
        grad2 = params2
        # print(f"prev:{torch.norm(prev_param)}|p1:{torch.norm(params1)}|p2:{torch.norm(params2)}")
        # print(f"g1:{torch.norm(grad1)}|g2:{torch.norm(grad2)}")
        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()


class Weight_diff(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Weight_diff, self).__init__(input_shape, num_classes, num_domains,
                                     hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.network = gradient_utils.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None
        self.optimizer_specific_state = [None] * num_domains

    def create_clone(self, device, n_domain):
        self.network_inner = gradient_utils.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

        self.network_specific = []
        self.optimizer_specific = []

        for i_domain in range(n_domain):
            self.network_specific.append(gradient_utils.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                            weights=self.network.state_dict()).to(device))
            self.optimizer_specific.append(torch.optim.Adam(
                self.network_specific[i_domain].parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            ))
            if self.optimizer_specific_state[i_domain] is not None:
                self.optimizer_specific[i_domain].load_state_dict(self.optimizer_specific_state[i_domain])

    # def fish(self, meta_weights, inner_weights, lr_meta):
    #     meta_weights = ParamDict(meta_weights)
    #     inner_weights = ParamDict(inner_weights)
    #     meta_weights += lr_meta * (inner_weights - meta_weights)
    #     return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device, self.num_domains)
        model_origin = copy.deepcopy(self.network)
        for i_domain, (x, y) in enumerate(minibatches):
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

            loss = F.cross_entropy(self.network_specific[i_domain](x), y)
            self.optimizer_specific[i_domain].zero_grad()
            loss.backward()
            self.optimizer_specific[i_domain].step()
            self.optimizer_specific_state[i_domain] = self.optimizer_specific[i_domain].state_dict()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        diff = [self.diff_weight(self.network_specific[i_domain], self.network) for i_domain in range(self.num_domains)]
        domain_diff_dict = {f"domain_{i}": value for i, value in enumerate(diff)}

        # Then, define the existing dictionary
        result_dict = {'loss': loss.item()}

        grad_norm = self.diff_weight(model_origin, self.network)
        grad_norm_dict = {f"grad_progress": grad_norm}

        # print(domain_diff_dict)
        # print(grad_norm_dict)

        result_dict.update(domain_diff_dict)
        result_dict.update(grad_norm_dict)
        return result_dict

    def predict(self, x):
        return self.network(x)


# class ParamDict(OrderedDict):
#     """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
#     A dictionary where the values are Tensors, meant to represent weights of
#     a model. This subclass lets you perform arithmetic on weights directly."""
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, *kwargs)
#
#     def _prototype(self, other, op):
#         if isinstance(other, Number):
#             return ParamDict({k: op(v, other) for k, v in self.items()})
#         elif isinstance(other, dict):
#             return ParamDict({k: op(self[k], other[k]) for k in self})
#         else:
#             raise NotImplementedError
#
#     def __add__(self, other):
#         return self._prototype(other, operator.add)
#
#     def __rmul__(self, other):
#         return self._prototype(other, operator.mul)
#
#     __mul__ = __rmul__
#
#     def __neg__(self):
#         return ParamDict({k: -v for k, v in self.items()})
#
#     def __rsub__(self, other):
#         # a- b := a + (-b)
#         return self.__add__(other.__neg__())
#
#     __sub__ = __rsub__
#
#     def __truediv__(self, other):
#         return self._prototype(other, operator.truediv)
