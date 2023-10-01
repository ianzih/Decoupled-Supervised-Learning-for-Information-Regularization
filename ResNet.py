import torch
import torch.nn as nn
from utils import conv_layer_bn, conv_1x1_bn, Flatten, ALComponent, ContrastiveLoss, PredSimLoss



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1, shortcut = None):
        super().__init__()
        self.conv1 = conv_layer_bn(in_channels, out_channels, nn.ReLU(inplace=True), stride, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.relu(out + identity)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, shortcut = None):
        super().__init__()
        self.conv1 = conv_1x1_bn(in_channels, out_channels, nn.ReLU(inplace=True), 1, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, nn.ReLU(inplace=True), stride, False)
        self.conv3 = conv_1x1_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.ReLU(inplace=True)

        # the shortcut output dimension is not the same with residual function
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.relu(out + identity)
        return out

class Resnet_block(nn.Module):
    def __init__(self, args):
        super(Resnet_block, self).__init__()
        self.args = args
        self.num_channel = 3
        self.layershape = 64
        self.shape = 64
        
    def _make_layer(self, block, out_channels, stride = 1, blocks = None):
        shortcut = None
        if stride != 1 or self.layershape != out_channels * block.expansion:
            shortcut = nn.Sequential(
                conv_1x1_bn(self.layershape, out_channels * block.expansion, stride = stride)
                )
        
        layers = []
        layers.append(block(self.layershape, out_channels, stride, shortcut))
        
        self.layershape = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.layershape, out_channels))
            
        return nn.Sequential(*layers)
    
    def _shape_mul_2(self):
        self.shape *= 2
        return self.shape
    
    def forward(self, x, y):
        if self.training:
            return self.train_step(x, y)
        else:
            return self.inference(x, y)
        
class resnet(Resnet_block):
    def __init__(self, args, block, layers):
        super(resnet, self).__init__(args)
        self.num_class = args.n_classes
        self.block = block
        self.layers = layers
        self.ce = nn.CrossEntropyLoss()
        
        self.conv1 = conv_layer_bn(self.num_channel, self.shape, nn.ReLU(inplace=True), stride = 1, kernel_size=3)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block = self.block, out_channels = self.shape, blocks = self.layers[0])
        self.layer2 = self._make_layer(block = self.block, out_channels = self._shape_mul_2(), stride = 2, blocks = self.layers[1])
        self.layer3 = self._make_layer(block = self.block, out_channels = self._shape_mul_2(), stride = 2, blocks = self.layers[2])
        self.layer4 = self._make_layer(block = self.block, out_channels = self._shape_mul_2(), stride = 2, blocks = self.layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.shape, self.num_class)
        
    def train_step(self, x , y):
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        return self.ce(output, y)
    
    def inference(self, x , y):
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        return output
        
      
class ResNet_AL_Component(ALComponent):
    def __init__(self, conv: nn.Module, flatten_size: int = 1024, hidden_size: int = 500, out_features: int = 10):
        g_function = nn.Sigmoid() 
        b_function = nn.Sigmoid()

        f = conv
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features), g_function)

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(ResNet_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca)
         
         
class resnet18_AL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.n_classes
        neurons = 500
        self.shape = 32
        
        conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)
        
        
        conv2_x = self._make_layer(64, 64, [1, 1])
        self.layer1 = ResNet_AL_Component(nn.Sequential(conv1, conv2_x), int(64 * self.shape * self.shape), neurons, self.num_classes)
        
        conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.layer2 = ResNet_AL_Component(conv3_x, int(128 * self.shape * self.shape), neurons, neurons)
        
        conv4_x = self._make_layer(128, 256, [2, 1])        
        self.shape /= 2
        self.layer3 = ResNet_AL_Component(conv4_x, int(256 * self.shape * self.shape), neurons, neurons)
        
        conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.layer4 = ResNet_AL_Component(conv5_x, int(512 * self.shape * self.shape), neurons, neurons)
        
    def forward(self, x, y):
        if self.training:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # generate one-hot encoding
            y_onehot = torch.zeros([len(y), self.num_classes]).to(device)
            for i in range(len(y)):
                y_onehot[i][y[i]] = 1.
            
            _s = x
            _t = y_onehot
            total_loss = {'f':[], 'b':[],'ae':[]}
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer1(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer2(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer3(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer4(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            return total_loss
        else:
            _s = x
            _s = self.layer1(_s, None)
            _s = self.layer2(_s, None)
            _s = self.layer3(_s, None)
            _t0 = self.layer4.bridge_forward(_s)
            _t0 = self.layer3(None, _t0)
            _t0 = self.layer2(None, _t0)
            _t0 = self.layer1(None, _t0)
            return _t0
    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)
    
    
class resnet18_SCPL(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.shape = 32
        self.conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)

        self.conv2_x = self._make_layer(64, 64, [1, 1])
        self.sclLoss1 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 64, shape = self.shape)
        self.conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.sclLoss2 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 128, shape = self.shape)
        self.conv4_x = self._make_layer(128, 256, [2, 1])
        self.shape /= 2
        self.sclLoss3 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 256, shape = self.shape)
        self.conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.sclLoss4 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.n_classes)

        self.ce = nn.CrossEntropyLoss()

    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x, y=None):
        loss = 0
        output = self.conv1(x)
        output = self.conv2_x(output)
        if self.training:
            loss += self.sclLoss1(output, y)
            output = output.detach()
        output = self.conv3_x(output)
        if self.training:
            loss += self.sclLoss2(output, y)
            output = output.detach()
        output = self.conv4_x(output)
        if self.training:
            loss += self.sclLoss3(output, y)
            output = output.detach()
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        if self.training:
            loss += self.sclLoss4(output, y)
            output = output.detach()
        
      
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output

        
class resnet18_PredSim(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.shape = 32
        self.conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)
        

        self.conv2_x = self._make_layer(64, 64, [1, 1])
        self.Loss1 = PredSimLoss(0.07, input_neurons = 2048, c_in = 64, shape = self.shape)
        self.conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.Loss2 = PredSimLoss(0.07, input_neurons = 2048, c_in = 128, shape = self.shape)
        self.conv4_x = self._make_layer(128, 256, [2, 1])
        self.shape /= 2
        self.Loss3 = PredSimLoss(0.07, input_neurons = 2048, c_in = 256, shape = self.shape)
        self.conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.Loss4 = PredSimLoss(0.07, input_neurons = 2048, c_in = 512, shape = self.shape)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.n_classes)
        self.ce = nn.CrossEntropyLoss()

    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        loss = 0
        output = self.conv1(x)
        output = self.conv2_x(output)
        if self.training:
            loss += self.Loss1(output, y)
            output = output.detach()
        output = self.conv3_x(output)
        if self.training:
            loss += self.Loss2(output, y)
            output = output.detach()
        output = self.conv4_x(output)
        if self.training:
            loss += self.Loss3(output, y)
            output = output.detach()
        output = self.conv5_x(output)

        if self.training:
            loss += self.Loss4(output, y)
            output = output.detach()
        output = self.avg_pool(output)

        output = output.view(output.size(0), -1)

        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output



