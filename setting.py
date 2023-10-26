from ResNet import *
from VGG import *
from vanillaCNN import *
from utils import *
from torchvision import transforms, datasets

def set_optim(model, optimal='LARS', args = None):
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

def set_model(name , args):
    if name == "VGG":
        model = VGG(args)
    elif name == "VGG_AL":
        model = VGG_AL(args)
    elif name == "VGG_SCPL":
        model = VGG_SCPL(args)
    elif name == "VGG_Research":
        model = VGG_Research(args)
    elif name == "VGG_PredSim":
        model = VGG_PredSim(args)
    elif name == "VGG_Research_Dynamic":
        model = VGG_Research_Dynamic(args)
    elif name == "VGG_Research_Adaptive":
        model = VGG_Research_Adaptive(args)
    elif name == "resnet18":
        model = resnet(args, BasicBlock, [2, 2, 2, 2])
    elif name == "resnet34":
        model = resnet(args, BasicBlock, [3, 4, 6, 3])
    elif name == "resnet50":
        model = resnet(args, Bottleneck, [3, 4, 6, 3])
    elif name == "resnet_AL":
        model = resnet18_AL(args)
    elif name == "resnet18_SCPL":
        model = resnet_SCPL(args, BasicBlock, [2, 2, 2, 2])
    elif name == "resnet34_SCPL":
        model = resnet_SCPL(args, BasicBlock, [3, 4, 6, 3])
    elif name == "resnet50_SCPL":
        model = resnet_SCPL(args, Bottleneck, [3, 4, 6, 3])
    elif name == "resnet_PredSim":
        model = resnet18_PredSim(args)
    elif name == "resnet18_Research":
        model = resnet_Research(args, BasicBlock, [2, 2, 2, 2])
    elif name == "resnet34_Research":
        model = resnet_Research(args, BasicBlock, [3, 4, 6, 3])
    elif name == "resnet50_Research":
        model = resnet_Research(args, Bottleneck, [3, 4, 6, 3])
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