import torch
from torchvision import transforms, datasets
from torch import optim
import configparser

import os
import math
import sys
import time
import argparse


from utils import AverageMeter, accuracy, TwoCropTransform, LARS, GetModelSizeVision, SetGPUDevices, Adjust_Learning_Rate, ResultRecorder, Calculate_GPUs_usage
from ResNet import resnet18, resnet18_AL, resnet18_SCPL, resnet18_PredSim
from VGG import VGG, VGG_AL, VGG_SCPL, VGG_PredSim, VGG_SCPL_Dynamic
from vanillaCNN import CNN, CNN_AL, CNN_SCPL, CNN_PredSim


def get_arguments():
    parser = argparse.ArgumentParser(description="Vision argu", add_help=False)

    # Model 
    parser.add_argument("--model", type = str, default = "VGG_SCPL_Dynamic", help = 'Model Name [CNN, CNN_AL, CNN_SCPL, CNN_PredSim, \
        VGG, VGG_AL, VGG_SCPL, VGG_PredSim, resnet, resnet_AL, resnet_SCPL, resnet_PredSim, VGG_SCPL_Dynamic]')
    
    # Dataset 
    parser.add_argument("--dataset", type = str, default = "cifar100", help = 'Dataset (cifar10, cifar100, tinyImageNet)')
    parser.add_argument("--aug_type", type = str, default = "strong", help = 'Dataset augmentation type(strong or basic)')
    parser.add_argument("--n_classes", type = int, default = 100, help = 'Number of Dataset classes)')
    
    # Optim
    parser.add_argument("--optimal", type = str, default = "LARS", help = 'Optimal Name (LARS, SGD, ADAM)')
    parser.add_argument('--epochs', type = int, default = 400, help = 'Number of training epochs')
    parser.add_argument('--train_bsz', type = int, default = 128, help = 'Batch size of training data')
    parser.add_argument('--test_bsz', type = int, default = 1024, help = 'Batch size of test data')
    parser.add_argument('--base_lr', type = float, default = 0.4, help = 'Initial learning rate')
    parser.add_argument('--end_lr', type = float, default = 0.004, help = 'Learning rate at the end of training')
    parser.add_argument('--max_steps', type = int, default = 2000, help = 'Learning step of training')
    parser.add_argument('--wd', type = float, default = 1e-4, help = 'Optim weight_decay')
    
    # Loss & GPU info.
    parser.add_argument("--localloss", type = str, default = "VICRIG", help = 'Defined local loss in each layer')
    parser.add_argument('--gpus', type=str, default="0", help=' ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, \
         e.g., \"0,1\". For single GPU models, only the first GPU ID will be used.')
    
    # other config
    parser.add_argument('--blockwise_total', type = int, default = 4, help = 'A total of model blockwise')
    parser.add_argument("--mlp", type = str, default = "2048-2048-2048", help = 'Size and number of layers of the MLP expander head')
    parser.add_argument("--merge", type = str, default="merge", help =' Decide whether to merge the classifier into the projector (merge, unmerge)')
    parser.add_argument("--showmodelsize", type = bool, default = False, help = 'Whether show model size (True, False)')
    parser.add_argument("--jsonfilepath", type = str, default="./modelresult/", help ='json file path for model result info.')
    parser.add_argument('--train_time', type = int, default = 1, help = 'Round Times of training step')
    
    return parser.parse_args()

args =  get_arguments()

def set_optim(model, optimal='LARS'):
    if optimal == 'LARS':
        optimizer = LARS(model.parameters(), lr = args.base_lr, weight_decay = args.wd, weight_decay_filter = LARS.exclude_bias_and_norm, 
        lars_adaptation_filter = LARS.exclude_bias_and_norm)
    elif optimal == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.base_lr)
    elif optimal == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.base_lr)
    
    return optimizer
    

def set_loader(dataset, train_bsz, test_bsz, augmentation_type):
    if dataset == "cifar10":
        n_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "cifar100":
        n_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "tinyImageNet":
        n_classes = 200
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))
    
    if dataset == "cifar10" or dataset == "cifar100":
        normalize = transforms.Normalize(mean=mean, std=std)
        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if dataset == "tinyImageNet":
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])

        
        weak_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])

    if augmentation_type == "basic":
        source_transform = weak_transform
        target_transform = None
    elif augmentation_type == "strong":
        source_transform = TwoCropTransform(weak_transform, strong_transform)
        target_transform = TwoCropTransform(None)
    else:
        raise ValueError("Augmentation type not supported: {}".format(augmentation_type))


    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root='./cifar10', transform=source_transform, target_transform = target_transform,  download=True)
        test_set = datasets.CIFAR10(root='./cifar10', train=False, transform=test_transform)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(root='./cifar100', transform=source_transform, target_transform = target_transform, download=True)
        test_set = datasets.CIFAR100(root='./cifar100', train=False, transform=test_transform)
    elif dataset == "tinyImageNet":
        train_set = datasets.ImageFolder('./tiny-imagenet-200/train', transform=source_transform)
        test_set = datasets.ImageFolder('./tiny-imagenet-200/val', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bsz, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsz, shuffle=False, pin_memory=True)
    

    return train_loader, test_loader, n_classes

def set_model(name):
    if name == "VGG":
        model = VGG(args)
    elif name == "VGG_AL":
        model = VGG_AL(args)
    elif name == "VGG_SCPL":
        model = VGG_SCPL(args)
    elif name == "VGG_SCPL_Dynamic":
        model = VGG_SCPL_Dynamic(args)
    elif name == "VGG_PredSim":
        model = VGG_PredSim(args)
    elif name == "resnet":
        model = resnet18(args)
    elif name == "resnet_AL":
        model = resnet18_AL(args)
    elif name == "resnet_SCPL":
        model = resnet18_SCPL(args)
    elif name == "resnet_PredSim":
        model = resnet18_PredSim(args)
    elif name == "CNN":
        model = CNN(args)
    elif name == "CNN_AL":
        model = CNN_AL(args)
    elif name == "CNN_SCPL":
        model = CNN_SCPL(args)
    elif name == "CNN_PredSim":
        model = CNN_PredSim(args)
    else:
        raise ValueError("Model not supported: {}".format(name))
    
    return model

def train(train_loader, model, optimizer, global_steps, epoch, aug_type, dataset):
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    classifier_acc = [AverageMeter() for _ in range(args.blockwise_total)]

    base = time.time()
    for step, (X, Y) in enumerate(train_loader):
        if aug_type == "strong":
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

        if type(loss) == dict:
            loss = sum(loss["f"]) + sum(loss["b"]) + sum(loss["ae"])
                            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)
        
        model.eval()
        with torch.no_grad():
            if args.model == "VGG_SCPL_Dynamic":
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
    if args.model == "VGG_SCPL_Dynamic":
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "DT {2:.3f}\t"
            "loss {3:.3f}\t"
            "acc {4:.3f}\t"
            "classifier acc {5}\t".format(epoch, (batch_time.avg)*len(train_loader), (data_time.avg)*len(train_loader), losses.avg, accs.avg, [format(classifier_acc[num].avg, ".3f") for num , _ in enumerate(classifier_acc)]))
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

            if args.model == "VGG_SCPL_Dynamic":
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
    if args.model == "VGG_SCPL_Dynamic":
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "Acc {2:.3f}\t"
            "classifier acc {3}\t".format(epoch, batch_time.avg*len(test_loader), accs.avg, [format(classifier_acc[num].avg, ".3f") for num , _ in enumerate(classifier_acc)]))
    else:
        print("Epoch: {0}\t"
            "Time {1:.3f}\t"
            "Acc {2:.3f}\t".format(epoch, batch_time.avg*len(test_loader), accs.avg))
        
    
    print("================================================")
    sys.stdout.flush()

    return accs.avg, [classifier_avg.avg for classifier_avg in classifier_acc], batch_time.avg*len(test_loader)


def main(time, result_recorder):
    best_acc = 0
    best_epoch = 0
    global_steps = 0
    
    train_loader, test_loader, args.n_classes = set_loader(args.dataset, args.train_bsz, args.test_bsz, args.aug_type)
    model = set_model(args.model).cuda() if torch.cuda.is_available() else set_model(args.model)
    optimizer = set_optim(model= model, optimal= args.optimal)
    GetModelSizeVision(model, train_loader, args)
    GPU_list = SetGPUDevices(args.gpus)
    
    args.max_steps = args.epochs * len(train_loader)
    print(args)
    for epoch in range(1, args.epochs + 1):
        lr = Adjust_Learning_Rate(optimizer, args.base_lr, args.end_lr, global_steps, args.max_steps)
        
        print("lr: {:.6f}".format(lr))
        loss, train_acc, global_steps, train_classifier_acc, train_time = train(train_loader, model, optimizer, global_steps, epoch, args.aug_type, args.dataset)
        test_acc,  test_classifier_acc, test_time= test(test_loader, model, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        result_recorder.epochresult(time, epoch, lr, train_acc, train_classifier_acc, loss, train_time, test_acc, test_classifier_acc, test_time)
        
    # Save Json Info.
    result_recorder.addinfo(time, best_acc, best_epoch, Calculate_GPUs_usage(GPU_list))   
        
    # Save Checkpoints    
    state = {
        "configs": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    save_files = os.path.join("./save_models/", "ckpt_last_{0}.pth".format(i))
    torch.save(state, save_files)
    
    del state
    print("Best accuracy: {:.2f}".format(best_acc))

if __name__ == '__main__':
    result_recorder = ResultRecorder(args)
    for i in range(args.train_time):
        main(i, result_recorder)
        
    result_recorder.save(args.jsonfilepath)
    









