import torch
import torch.nn as nn
import math
import os
import json
import torch.nn.functional as F
from torch import optim
from utils.vision_utils import *


class ALComponent(nn.Module):
    """
        Base class of a single associated learning block
        
        f: forward function
        g: autoencoder function
        b: bridge function
        inv: inverse function of autoencoder
        cb: creterion of bridge function
        ca: creterion of autoencoder
        cf: creterion of forward function
    """
    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        b: nn.Module,
        inv: nn.Module,
        cf: nn.Module,
        cb: nn.Module,
        ca: nn.Module
    )->None:
        super(ALComponent, self).__init__()
        self.f = f
        self.g = g
        self.b = b
        self.inv = inv
        self.cb = cb
        self.ca = ca
        self.cf = cf
    
    def forward(self, x=None, y=None, label=None):
        if self.training:
            s = self.f(x)
            #loss_f = 100 * self.cf(s, label)
            loss_f = 0
            s0 = self.b(s)
            t = self.g(y)
            t0 = self.inv(t)
            
            loss_b = self.cb(s0, t.detach()) # contrastive loss
            loss_ae = self.ca(t0, y)
            return s.detach(), t.detach(), loss_f, loss_b, loss_ae
        else:
            if y == None:
                s = self.f(x)
                return s
            else:
                t0 = self.inv(y)
                return t0
        
    # for bridge block inference
    def bridge_forward(self, x):
        s = self.f(x)
        s0 = self.b(s)
        t0 = self.inv(s0)
        return t0

class VICRIG(nn.Module):
    def __init__(self, input_channel = 256, shape = 32 , args = None, activation = nn.ReLU()):
        super(VICRIG, self).__init__()
        self.n_class = args.n_classes
        self.num_features = int(args.mlp.split("-")[-1])
        if args.task != "nlp":
            self.projector = Projector(args, int(input_channel * shape * shape), activation)
        else:
            self.projector = nn.Identity()
        
    def forward(self, x, label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss = 0
        x = self.projector(x)
        
        if self.training:
            nor_x =  nn.functional.normalize(x)
            batch_size = label.shape[0]
            
            # covar
            x_mean = nor_x - nor_x.mean(dim=0)
            cov_x = (x_mean.T @ x_mean) / (batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow(2).sum().div(self.num_features)

            # invar
            target_onehot = to_one_hot(label.to(device), self.n_class)
            target_sm = similarity_matrix(target_onehot)
            x_sm = similarity_matrix(nor_x)
            invar_loss = F.mse_loss(target_sm.to(device), x_sm.to(device))
            
            # var
            x_mean = nor_x - nor_x.mean(dim=0)
            std_label = torch.sqrt(x_mean.var(dim=0) + 0.0000001) 
            var_loss = torch.mean(F.relu(1 - std_label)) / (batch_size - 1)
            
            loss = ( var_loss * 1.0 + invar_loss * 1.0 + cov_loss * 1.0)

        return loss, x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, mid_neurons = 512, out_neurons = 1024, input_channel = 256, shape = 32, args = None ,activation = nn.ReLU()):
        super(ContrastiveLoss, self).__init__()
        input_neurons = int(input_channel * shape * shape)
        if args.task != "nlp":
            self.linear = nn.Sequential(Flatten(), nn.Linear(input_neurons, mid_neurons), activation, nn.Linear(mid_neurons, out_neurons))
        else:
            self.linear = nn.Identity()
        self.temperature = temperature
    
    def forward(self, x, label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = self.linear(x)
        x =  nn.functional.normalize(x)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).float().to(device)
        # 製造對角是0 其他1   # mask * denom_mask (去除自己)
        denom_mask = torch.scatter(torch.ones_like(mask).to(device), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        # 任何data之前 相似度 (外積)
        logits = torch.div(torch.matmul(x, x.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # neg_mask = torch.ones_like(mask).to(device) - mask
        # denom = torch.exp(logits) * neg_mask
        
        # 算出除了自己以外的 (contrastive分母)
        denom = torch.exp(logits) * denom_mask
        
        # 分子 / 分母
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        
        # 取出自己相符合的正對  這裡的mask包含自己  但SCL code 不包含自己
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        
        loss = -loss
        loss = loss.view(1, batch_size).mean()

        return loss

class PredSimLoss(nn.Module):
    def __init__(self, temperature = 0.1, input_neurons = 2048, c_in = 256, shape = 32):
        super().__init__()
        num_classes = 200
        self.conv_loss = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, bias=False)
        self.decoder_y = nn.Linear(input_neurons, num_classes)
        # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
        ks_h, ks_w = 1, 1
        dim_out_h, dim_out_w = shape, shape
        dim_in_decoder = c_in*dim_out_h*dim_out_w
        while dim_in_decoder > input_neurons and ks_h < shape:
            ks_h*=2
            dim_out_h = math.ceil(shape / ks_h)
            dim_in_decoder = c_in*dim_out_h*dim_out_w
            if dim_in_decoder > input_neurons:
               ks_w*=2
               dim_out_w = math.ceil(shape / ks_w)
               dim_in_decoder = c_in*dim_out_h*dim_out_w 
        if ks_h > 1 or ks_w > 1:
            pad_h = (ks_h * (dim_out_h - shape // ks_h)) // 2
            pad_w = (ks_w * (dim_out_w - shape // ks_w)) // 2
            self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(0, 0))
        else:
            self.avg_pool = None
    def forward(self, h, y):
        y_onehot = nn.functional.one_hot(y, num_classes=200).float()
        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)
        
        if self.avg_pool is not None:
            h = self.avg_pool(h)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))
        
        Ry = similarity_matrix(y_onehot).detach()
        loss_pred = (1-0.99) * F.cross_entropy(y_hat_local,  y.detach())
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        loss = loss_pred + loss_sim
        
        return loss
    
class Layer_Classifier(nn.Module):
    def __init__(self, input_channel = 2048, args = None, activation = nn.ReLU()):
        super().__init__()
        self.n_class = args.n_classes
        self.num_features = int(args.mlp.split("-")[-1])
        self.classifier = Classifier(args, input_channel, self.num_features, activation)
    
    def forward(self, x):
        output = self.classifier(x)
        
        return output
        
def Projector(args, input_channel, activation = nn.ReLU()):
    mlp_spec = f"{input_channel}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    if args.task == "vision":
        layers.append(Flatten())
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if args.task == "vision":
            layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(activation)
    layers.append(nn.Linear(f[-2], f[-1]))
    return nn.Sequential(*layers)

def Classifier(args, input_channel, final_channels, activation = nn.ReLU()):
    mlp_spec = f"{input_channel}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    if args.merge == 'unmerge':
        if args.task == "vision":
            layers.append(Flatten())
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            if args.task == "vision":
                layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(f[-2], args.n_classes))
    else:
        if args.task == "vision":
            layers.append(nn.BatchNorm1d(final_channels))
        layers.append(activation)
        layers.append(nn.Linear(final_channels, args.n_classes))
    return nn.Sequential(*layers)

def to_one_hot(y, n_dims=None):
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def similarity_matrix(x , no_similarity_std = False):
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def Set_Local_Loss(args, input_channel, shape, activation = nn.ReLU()):
    if args.localloss == 'VICRIG':
        return VICRIG(input_channel, shape, args, activation)
    elif args.localloss == 'contrastive':
        return ContrastiveLoss(input_channel = input_channel, shape = shape, args = args, activation = nn.ReLU())


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
    
def accuracy(output, target):
    with torch.no_grad():
        bsz = target.shape[0]
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = correct[0].view(-1).float().sum(0, keepdim=True).mul_(100 / bsz)
        return acc
    

# Learning rate setting
def Adjust_Learning_Rate(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    set_lr(optimizer, lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

# LARS Optimizer
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
                
    def exclude_bias_and_norm(p):
        return p.ndim == 1
    
def GetModelSize(model, train_loader, args):
    try:
        from torchinfo import summary
    except Exception as E:
        print("You need to install python package - \"pip install torchinfo==1.8.0\"")
        os._exit(0)
    
    if args.showmodelsize == True:
        for _, (X, Y) in enumerate(train_loader): 
            if args.task == "vision":
                if args.aug_type == "strong":
                    if args.dataset == "cifar10" or args.dataset == "cifar100":
                        X = torch.cat(X).cuda(non_blocking=True)
                        Y = torch.cat(Y).cuda(non_blocking=True)
                    else:
                        X = torch.cat(X).cuda(non_blocking=True)
                        Y = torch.cat([Y, Y]).cuda(non_blocking=True)
            else:
                X = X.cuda(non_blocking=True)
                Y = Y.cuda(non_blocking=True)

            model.train()
            summary(model, depth = 10, input_data = [X,Y], batch_dim = args.train_bsz, verbose = 1)

            break

def SetGPUDevices(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    return list(map(str, gpu_id.split(",")))

def get_gpu_info(gpu_id):
    gpu_info = torch.cuda.get_device_properties(gpu_id)
    gpu_name = gpu_info.name
    gpu_cuda_ver = "{}.{}_{}".format(gpu_info.major, gpu_info.minor, torch.version.cuda) 
    total_m = gpu_info.total_memory
    reserved_m = torch.cuda.memory_reserved(gpu_id)
    allocated_m = torch.cuda.memory_allocated(gpu_id)
    free_m = reserved_m-allocated_m  
    
    gpuinfo = {"id": gpu_id, "name": gpu_name, "cuda": gpu_cuda_ver, "total": total_m,
        "reserved": reserved_m, "allocated": allocated_m, "free": free_m}
    
    return gpuinfo

def Calculate_GPUs_usage(gpu_ids:list()):
    total_all_m = 0
    reserved_all_m = 0
    gpus_info = list()

    for idx in range(len(set(gpu_ids))):
        gpu_info = get_gpu_info(int(gpu_ids[idx]))
        total_all_m = gpu_info['total'] + total_all_m
        reserved_all_m = gpu_info['reserved'] + reserved_all_m
        gpus_info.append(gpu_info)

    return {"total_all_m":total_all_m, "reserved_all_m":reserved_all_m, "gpus_info": gpus_info}

class ResultRecorder(object):
    def __init__(self, args = None):
        self.args = args
        self.hyperparam = dict()
        self.modelresult = dict()
        self.epochtrainresult = dict()
        
        self._initjsonparam()
    
    def _initjsonparam(self):
        self.times = 'Round Time' 
        self.best_test_acc = 'Best Test Acc'
        self.best_test_epoch = 'Best Test Epoch'
        self.train_last_acc = 'Train Last Acc'
        self.lr = 'Learning Rate'
        self.epoch = 'Epoch'
        self.train_last_acc = 'Train Last Acc'
        self.gpus_info = 'GPU Info.'
        self.train_classifier_acc =  'Train Classifier Acc'
        self.train_loss = 'Train Loss'
        self.train_time = 'Train Time'
        self.test_last_acc = 'Test Last Acc'
        self.test_classifier_acc =  'Test Classifier Acc'
        self.test_time = 'Test Time'
        self.best_acc_layer = "Best_ACC_Layer"
    
    def addinfo(self, times, best_test_acc, best_test_epoch, gpus_info, best_acc_layer):
        self.modelresult[times] = {
            self.times: times,
            self.best_test_acc: best_test_acc,
            self.best_acc_layer: best_acc_layer,
            self.best_test_epoch: best_test_epoch,
            self.gpus_info: gpus_info
        }
        
    def epochresult(self, roundtime, epoch, lr, trainacc_L, trainacc_C, loss, traintime, testacc_L, testacc_C, testtime):
        self.epochtrainresult.setdefault(roundtime, {})
        
        self.epochtrainresult[roundtime][epoch] = {
            self.epoch: epoch,
            self.lr: lr,
            self.train_last_acc: trainacc_L,
            self.train_classifier_acc: trainacc_C,
            self.train_loss: loss,
            self.train_time: traintime,
            self.test_last_acc: testacc_L,
            self.test_classifier_acc: testacc_C,
            self.test_time: testtime
        }
    
    def _makefinalresult(self):
        if self.args.task == "nlp":
            self.args.word_vec = "save"
        self.hyperparam = vars(self.args)
        final_result = {
            'Model Hyperparam' : self.hyperparam,
            'Model Best Result Value' : self.modelresult,
            'Round Time Result Summary' : self.epochtrainresult,
        }
        return final_result
    
    def save(self, path):     
        import datetime
        if not os.path.exists(path):
            os.makedirs(path)
        
        filename = '(' + self.args.dataset + ')_(' + self.args.model + ')_(' + str(self.args.base_lr) + \
            ')_(' +  self.args.mlp + ')'
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = path + filename + '_d(' + now + ').json'

        with open(path, 'w', encoding='utf-8') as json_f:
            json_f.write(json.dumps(self._makefinalresult(), indent = 4)) 
        
        print("[INFO] Save results, file name: {}".format(path))
        