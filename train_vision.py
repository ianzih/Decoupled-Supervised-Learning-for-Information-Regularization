import torch

import os
import math
import sys
import time
import argparse
import numpy as np

from utils.utils import *
from utils.vision_utils import *
from vision.setting import *


def get_arguments():
    parser = argparse.ArgumentParser(description="Vision argu", add_help=False)

    # Model 
    parser.add_argument("--task", type = str, default = "vision", help = 'task')
    parser.add_argument("--model", type = str, default = "VGG_DeInfoReg", help = 'Model Name [VGG, VGG_AL, VGG_SCPL \
        , VGG_DeInfoReg, resnet18, resnet18_AL, resnet18_SCPL,  resnet18_DeInfoReg, VGG_DeInfoReg_Adaptive]')
    
    # Dataset 
    parser.add_argument("--dataset", type = str, default = "cifar100", help = 'Dataset (cifar10, cifar100, tinyImageNet)')
    parser.add_argument("--aug_type", type = str, default = "basic", help = 'Dataset augmentation type(SCPL or basic)')
    parser.add_argument("--n_classes", type = int, default = 100, help = 'Number of Dataset classes)')
    
    # Optim
    parser.add_argument("--optimal", type = str, default = "LARS", help = 'Optimal Name (LARS, SGD, ADAM)')
    parser.add_argument('--epochs', type = int, default = 200, help = 'Number of training epochs')
    parser.add_argument('--train_bsz', type = int, default = 128, help = 'Batch size of training data')
    parser.add_argument('--test_bsz', type = int, default = 1024, help = 'Batch size of test data')
    parser.add_argument('--base_lr', type = float, default = 0.2, help = 'Initial learning rate')
    parser.add_argument('--end_lr', type = float, default = 0.002, help = 'Learning rate at the end of training')
    parser.add_argument('--max_steps', type = int, default = 2000, help = 'Learning step of training')
    parser.add_argument('--wd', type = float, default = 1e-4, help = 'Optim weight_decay')
    
    # Loss & GPU info.
    parser.add_argument("--localloss", type = str, default = "DeInfoReg", help = 'Defined local loss in each layer')
    parser.add_argument('--gpus', type=str, default="0", help=' ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, \
         e.g., \"0,1\". For single GPU models, only the first GPU ID will be used.')
    
    # other config
    parser.add_argument('--blockwise_total', type = int, default = 4, help = 'A total of model blockwise')
    parser.add_argument("--mlp", type = str, default = "2048-2048-2048", help = 'Size and number of layers of the MLP expander head')
    parser.add_argument("--merge", type = str, default="merge", help =' Decide whether to merge the classifier into the projector (merge, unmerge)')
    parser.add_argument("--showmodelsize", type = bool, default = False, help = 'Whether show model size (True, False)')
    parser.add_argument("--jsonfilepath", type = str, default="./modelresult/", help ='json file path for model result info.')
    parser.add_argument('--train_time', type = int, default = 1, help = 'Round Times of training step')
    parser.add_argument('--trigger_epoch', type=str, default="20,40", help='This augment is only use in dynamic model. e.g., layer = 4 \"20,40,60\"')
    parser.add_argument('--epoch_now', type = int, default = 1, help = 'Number of epoch now')
    parser.add_argument('--patiencethreshold', type = int, default = 1, help = 'threshold of inference adaptive patience count')
    parser.add_argument('--cosinesimthreshold', type = float, default = 0.8, help = 'threshold of inference adaptive cosine similarity')
    parser.add_argument('--noise_rate', type=float, default = 0.0, help='Noise rate of labels in training dataset (default is 0 for no noise).')
    
    return parser.parse_args()

args =  get_arguments()

def train(train_loader, model, optimizer, global_steps, epoch, aug_type, dataset):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    classifier_acc = [AverageMeter() for _ in range(args.blockwise_total)]

    base = time.time()
    for step, (X, Y) in enumerate(train_loader):
        if aug_type == "SCPL":
            if dataset == "cifar10" or dataset == "cifar100":
                X = torch.cat(X)
                Y = torch.cat(Y)
            else:
                X = torch.cat(X)
                Y = torch.cat([Y, Y])

        model.train()
        data_time.update(time.time()-base)

        if torch.cuda.is_available():
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
        bsz = Y.shape[0]

        global_steps += 1

        loss = model(X, Y)
                            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)
        
        model.eval()
        with torch.no_grad():
            if args.model in ["VGG_DeInfoReg", "VGG_DeInfoReg_Dynamic", "resnet18_DeInfoReg", "resnet34_DeInfoReg", "resnet50_DeInfoReg"]:
                output , classifier_output = model(X, Y)
                classifier_output_list = [num for val in classifier_output.values() for num in val]
                for num , val in enumerate(classifier_output_list):
                    acc = accuracy(val, Y)
                    classifier_acc[num].update(acc.item(), bsz)
            else:
                output = model(X, Y)
                acc = accuracy(output, Y)
                accs.update(acc.item(), bsz)

        batch_time.update(time.time()-base)
        base = time.time()
    
    # print info
    if args.model in ["VGG_DeInfoReg", "VGG_DeInfoReg_Dynamic", "resnet18_DeInfoReg", "resnet34_DeInfoReg", "resnet50_DeInfoReg"]:
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "DT {2:.3f}\t"
            "loss {3:.3f}\t"
            "Max acc {4:.3f}\t"
            "classifier acc {5}\t".format(epoch, (batch_time.avg)*len(train_loader), (data_time.avg)*len(train_loader), 
                                          losses.avg, max(classifier_acc, key=lambda x: x.avg).avg, [format(classifier_acc[num].avg, ".3f") for num , _ in enumerate(classifier_acc)]))
        sys.stdout.flush()
        return losses.avg, max(classifier_acc, key=lambda x: x.avg).avg, global_steps, [classifier_avg.avg for classifier_avg in classifier_acc], batch_time.avg*len(train_loader)
    else:
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "DT {2:.3f}\t"
            "loss {3:.3f}\t"
            "acc {4:.3f}\t".format(epoch, (batch_time.avg)*len(train_loader), (data_time.avg)*len(train_loader), losses.avg, accs.avg))
        sys.stdout.flush()
        return losses.avg, accs.avg, global_steps, [classifier_avg.avg for classifier_avg in classifier_acc], batch_time.avg*len(train_loader)



def test(test_loader, model, epoch):
    model.eval()

    batch_time = AverageMeter()
    accs = AverageMeter()
    classifier_acc = [AverageMeter() for _ in range(args.blockwise_total)]

    with torch.no_grad():
        base = time.time()
        for step, (X, Y) in enumerate(test_loader):
 
            if torch.cuda.is_available():
                X = X.cuda(non_blocking=True)
                Y = Y.cuda(non_blocking=True)
            bsz = Y.shape[0]

            if args.model in ["VGG_DeInfoReg", "VGG_DeInfoReg_Dynamic", "resnet18_DeInfoReg", "resnet34_DeInfoReg", "resnet50_DeInfoReg"]:
                output , classifier_output = model(X, Y)
                classifier_output_list = [num for val in classifier_output.values() for num in val]
                for num , val in enumerate(classifier_output_list):
                    acc = accuracy(val, Y)
                    classifier_acc[num].update(acc.item(), bsz)
            else:
                output = model(X, Y)
                acc = accuracy(output, Y)
                accs.update(acc.item(), bsz) 

            batch_time.update(time.time()-base)
            base = time.time()

    # print info
    if args.model in ["VGG_DeInfoReg", "VGG_DeInfoReg_Dynamic", "resnet18_DeInfoReg", "resnet34_DeInfoReg", "resnet50_DeInfoReg"]:
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "Acc {2:.3f}\t"
            "classifier acc {3}\t".format(epoch, batch_time.avg*len(test_loader), max(classifier_acc, key=lambda x: x.avg).avg, [format(classifier_acc[num].avg, ".3f") for num , _ in enumerate(classifier_acc)]))
        print("================================================")
        sys.stdout.flush()
        return max(classifier_acc, key=lambda x: x.avg).avg, [classifier_avg.avg for classifier_avg in classifier_acc], batch_time.avg*len(test_loader)
    else:
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "Acc {2:.3f}\t".format(epoch, batch_time.avg*len(test_loader), accs.avg))
        print("================================================")
        sys.stdout.flush()
        return accs.avg, [classifier_avg.avg for classifier_avg in classifier_acc], batch_time.avg*len(test_loader)  


def main(time, result_recorder):
    best_acc = 0
    best_acc_layer = 0 
    best_epoch = 0
    global_steps = 0
    
    GPU_list = SetGPUDevices(args.gpus)
    train_loader, test_loader, args.n_classes = set_loader(args.dataset, args.train_bsz, args.test_bsz, args.aug_type, args)
    model = set_model(args.model, args).cuda() if torch.cuda.is_available() else set_model(args.model, args)
    optimizer = set_optim(model= model, optimal= args.optimal, args = args)
    GetModelSize(model, train_loader, args)
    
    args.max_steps = args.epochs * len(train_loader)
    print(args)
    for epoch in range(1, args.epochs + 1):
        lr = Adjust_Learning_Rate(optimizer, args.base_lr, args.end_lr, global_steps, args.max_steps)
        args.epoch_now = epoch
        print("lr: {:.6f}".format(lr))
        loss, train_acc, global_steps, train_classifier_acc, train_time = train(train_loader, model, optimizer, global_steps, epoch, args.aug_type, args.dataset)
        test_acc,  test_classifier_acc, test_time= test(test_loader, model, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            bestmodel = model
            bestoptimizer = optimizer
            best_acc_layer = np.argmax(test_classifier_acc)
        result_recorder.epochresult(time, epoch, lr, train_acc, train_classifier_acc, loss, train_time, test_acc, test_classifier_acc, test_time)
        
    # Save Json Info.
    result_recorder.addinfo(time, best_acc, best_epoch, Calculate_GPUs_usage(GPU_list), str(best_acc_layer))   
    result_recorder.save(args.jsonfilepath)
        
    # Save Checkpoints    
    state = { "configs": args, "model": bestmodel.state_dict(), "optimizer": bestoptimizer.state_dict(), "epoch": best_epoch}
    if not os.path.exists("./save_nlp_models/"):
        os.makedirs("./save_nlp_models/")
    save_files = os.path.join("./save_nlp_models/", "ckpt_last_{0}.pth".format(i))
    torch.save(state, save_files)
    
    del state
    print("Best accuracy: {:.2f}".format(best_acc), "Best epoch: {:.2f}".format(best_epoch))

if __name__ == '__main__':
    result_recorder = ResultRecorder(args)
    for i in range(args.train_time):
        main(i, result_recorder)
        