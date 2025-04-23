import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from utils.data_utils import read_client_data
from utils.dlg import DLG

# from torch.utils.tensorboard import SummaryWriter
import wandb

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        # self.global_learning_rate_decay = args.global_learning_rate_decay
        # self.mmt = args.momentum
        # self.ss = args.step_size
        # self.gam = args.gamma
        self.model_str = args.model_str


        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_gradients = []
        self.grads = []
        self.model_subtraction = copy.deepcopy(args.model)

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval  # this is for what ??
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.args = args
        self.angle_ug = 0
        self.angle_uv = 0
        self.angle_neg_uv = 0
        self.angle_neg_ratio = 0

        self.grads_angle_value = 0
        

        if self.args.log:
            args.run_name = f"{args.algorithm}__{args.dataset}__{args.num_clients}__{int(time.time())}"

            self.current_round = 0
            self.domain_current_round = 0
            self.save_dir = f"runs/{args.run_name}"

            wandb.init(
                project="FL-DG",
                entity="scalemind",
                config=args,
                name=args.run_name,
                force=True
            )


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        if self.args.domain_training:
            self.current_num_join_clients -= 1
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # After update the self.global model then the for loop begins again and with this function
    # and the client get the global model parameter again

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        # print(len(active_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_gradients = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                # print(tot_samples)
                # print("client train_samples", client.train_samples)
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)  # train_samples = len(train_data)
                self.uploaded_models.append(client.model)
                self.uploaded_gradients.append(client.store_gradient_model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        """     
        for i, model in enumerate(self.uploaded_models, 1):
            print(f"client", i)
            print(f"model", model)
            for name, param in model.named_parameters(): 
                print(f"name", name)
                print(f"param", param)
        """

    def receive_grads(self):

        self.grads = copy.deepcopy(self.uploaded_models)
        # This for copy the list to store all the gradient update value

        for model in self.grads:
            for param in model.parameters():
                param.data.zero_()

        for grad_model, local_model in zip(self.grads, self.uploaded_models):
            for grad_param, local_param, global_param in zip(grad_model.parameters(), local_model.parameters(),
                                                             self.global_model.parameters()):
                grad_param.data = local_param.data - global_param.data

        for w, client_model in zip(self.uploaded_weights, self.grads):
            self.mul_params(w, client_model)

    def mul_params(self, w, client_model):
        for param in client_model.parameters():
            param.data = param.data.clone() * w

    def diff_weight(self, model1, model2):
        params1 = [p.data for p in model1.parameters()]
        params2 = [p.data for p in model2.parameters()]

        diff_norms = [torch.norm(p1 - p2, p='fro') for p1, p2 in zip(params1, params2)]

        total_diff_norm = torch.sum(torch.stack(diff_norms))
        return total_diff_norm.item()

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        grad1 = params1 - prev_param
        grad2 = params2

        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()

    def cosine_similarity(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        cos_sim = torch.dot(params1, params2) / (torch.norm(params1) * torch.norm(params2))
        return cos_sim.item()

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def model_aggregate_new(self):
        self.model_subtraction = copy.deepcopy(self.global_model)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        for old_param, new_param in zip(self.model_subtraction.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data - old_param.data

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            # print(client_param.data)
            server_param.data += client_param.data.clone() * w
    # this function will update the global model
    # need to see where the global model go

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            # algo = (algo + "_" + self.model_str + "_" + str(self.batch_size) + "_" + str(self.global_rounds) + "_" + str(self.cagrad_rounds)
            #         + "_" + str(self.ca_lr) + "_" + str(self.mmt) + "_" + str(self.ss) + "_" + str(self.gam))
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))
    
    def test_domain_metrics(self):
        self.fine_tuning_new_clients()
        if self.args.test_full_data:
            return self.test_metrics_full_data_new_clients()
        return self.test_metrics_new_clients()
    
    def test_metrics(self):
        # if self.eval_new_clients and self.num_new_clients > 0:
        #     self.fine_tuning_new_clients()
        #     return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        # print(f"ids:{ids}")
        # print(f"num_samples:{num_samples}")
        # print(f"tot_correct:{tot_correct}")
        # print(f"tot_auc:{tot_auc}")

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            # c.client_model
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    
    def domain_evaluate(self, acc=None, loss=None):
        print("domain eval")
        stats = self.test_domain_metrics()
        print(stats)
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        print("Domain Test Accurancy: {:.4f}".format(test_acc))
        print("Domain Test AUC: {:.4f}".format(test_auc))

        if self.args.log:

            wandb.log({"charts/domain_id_{}_test_acc".format(stats[0]): test_acc}, step=self.domain_current_round)

            self.domain_current_round += 1

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        angle_ug = self.angle_ug
        angle_uv = self.angle_uv
        angle_neg_uv = self.angle_neg_uv
        angle_neg_ratio = self.angle_neg_ratio

        grads_angle_value = self.grads_angle_value
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

        test_acc_std = np.std(accs).item()
        test_auc_std = np.std(aucs).item()
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Mean_Angle_Value_Compare_Global: {:.4f}".format(angle_ug))
        print("Mean_Angle_Among_Users: {:.4f}".format(angle_uv))
        print("Mean_Angle_Among_Conflicted_Users: {:.4f}".format(angle_neg_uv))
        print("Conflicted_Users_Ratio: {:.4f}".format(angle_neg_ratio))

        if self.args.log:
            for i in range(len(self.clients)):
                wandb.log({"charts/test_acc_id_{}".format(stats[0][i]):(stats[2][i]/stats[1][i])}, step=self.current_round)

            # self.writer.add_scalar("charts/train_loss", train_loss, self.current_round)
            wandb.log({"charts/train_loss": train_loss}, step=self.current_round)

            # self.writer.add_scalar("charts/test_acc", test_acc, self.current_round)
            wandb.log({"charts/test_acc": test_acc}, step=self.current_round)

            # self.writer.add_scalar("charts/test_auc", test_auc, self.current_round)
            wandb.log({"charts/test_auc": test_auc}, step=self.current_round)

            # self.writer.add_scalar("charts/test_acc_std", test_acc_std, self.current_round)
            wandb.log({"charts/test_acc_std": test_acc_std}, step=self.current_round)

            # self.writer.add_scalar("charts/test_auc_std", test_auc_std, self.current_round)
            wandb.log({"charts/test_auc_std": test_auc_std}, step=self.current_round)

            # self.writer.add_scalar("charts/angle_value", angle_ug, self.current_round)
            wandb.log({"charts/angle_value": angle_ug}, step=self.current_round)

            # self.writer.add_scalar("charts/user_angle_value", angle_uv, self.current_round)
            wandb.log({"charts/user_angle_value": angle_uv}, step=self.current_round)

            # self.writer.add_scalar("charts/user_neg_angle", angle_neg_uv, self.current_round)
            wandb.log({"charts/user_neg_angle": angle_neg_uv}, step=self.current_round)

            # self.writer.add_scalar("charts/neg_user_ratio", angle_neg_ratio, self.current_round)
            wandb.log({"charts/neg_user_ratio": angle_neg_ratio}, step=self.current_round)

            # self.writer.add_scalar("charts/grads_angle_value", grads_angle_value, self.current_round)
            # wandb.log({"charts/grads_angle_value", grads_angle_value}, step=self.current_round)

            self.current_round += 1

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        i = self.remove_domain
        # for i in range(self.num_clients, self.num_clients + self.num_new_clients):
        train_data = read_client_data(self.dataset, i, is_train=True)
        test_data = read_client_data(self.dataset, i, is_train=False)
        # train_data = read_client_data(self.dataset, i, self.args.noniid, self.args.balance, self.args.alpha_dirich,
        #                               is_train=True, num_clients=self.num_clients)
        # test_data = read_client_data(self.dataset, i, self.args.noniid, self.args.balance, self.args.alpha_dirich,
        #                              is_train=False, num_clients=self.num_clients)
        client = clientObj(self.args, 
                        id=i, 
                        train_samples=len(train_data), 
                        test_samples=len(test_data), 
                        train_slow=False, 
                        send_slow=False)
        self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            # client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            celoss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = celoss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def set_new_client_domain(self):
        print("\nlen",len(self.new_clients))
        for client in self.new_clients:
            client.set_parameters(self.global_model)

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    # def test_metrics_full_data_new_clients(self):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
    #     for c in self.new_clients:
    #         ct, ns, auc = c.test_full_metrics()
    #         tot_correct.append(ct*1.0)
    #         tot_auc.append(auc*ns)
    #         num_samples.append(ns)

    #     ids = [c.id for c in self.new_clients]
    #     print(tot_correct)

    #     return ids, num_samples, tot_correct, tot_auc
