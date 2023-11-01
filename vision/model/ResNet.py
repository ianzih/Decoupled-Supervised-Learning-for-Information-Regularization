import torch
import torch.nn as nn
from utils import conv_layer_bn, conv_1x1_bn, Flatten, ALComponent, ContrastiveLoss, PredSimLoss, Layer_Classifier, Set_Local_Loss



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
        self.conv3 = conv_1x1_bn(out_channels, out_channels * Bottleneck.expansion, None, 1, False)
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
        self.dim = 64
        self.shape = 32
        
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
    
    def _dim_mul_2(self):
        self.dim *= 2
        return self.dim
    
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape
    
    def forward(self, x, y):
        if self.training:
            return self.train_step(x, y)
        else:
            return self.inference(x, y)
        
class resnet(Resnet_block):
    def __init__(self, args, block, layers):
        super(resnet, self).__init__(args)
        self.num_classes = args.n_classes
        self.block = block
        self.layers = layers
        self.ce = nn.CrossEntropyLoss()
        
        self.conv1 = conv_layer_bn(self.num_channel, self.dim, nn.ReLU(inplace=True), stride = 1, kernel_size=3)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block = self.block, out_channels = self.dim, blocks = self.layers[0])
        self._shape_div_2()
        self.layer2 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[1])
        self._shape_div_2()
        self.layer3 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[2])
        self._shape_div_2()
        self.layer4 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[3])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(self.dim * block.expansion, self.num_classes)
        self.fc = nn.Sequential(Flatten(), nn.Linear(int(self.dim * block.expansion *  self.shape * self.shape), 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, self.num_classes))
        
    def train_step(self, x , y):
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        # output = self.avgpool(output)
        # output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        return self.ce(output, y)
    
    def inference(self, x , y):
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        # output = self.avgpool(output)
        # output = output.view(output.size(0), -1)
        
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
    

class resnet_SCPL(Resnet_block):
    def __init__(self, args, block, layers):
        super(resnet_SCPL, self).__init__(args)
        self.num_classes = args.n_classes
        self.block = block
        self.layers = layers
        self.ce = nn.CrossEntropyLoss()
        
        self.conv1 = conv_layer_bn(self.num_channel, self.dim, nn.ReLU(inplace=True), stride = 1, kernel_size=3)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block = self.block, out_channels = self.dim, blocks = self.layers[0])
        self.loss1 =  ContrastiveLoss(0.1, input_channel = self.dim * block.expansion, shape = self.shape)
        self.layer2 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[1])
        self.loss2 =  ContrastiveLoss(0.1, input_channel = self.dim * block.expansion, shape = self._shape_div_2())
        self.layer3 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[2])
        self.loss3 =  ContrastiveLoss(0.1, input_channel = self.dim * block.expansion, shape = self._shape_div_2())
        self.layer4 = self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[3])
        self.loss4 =  ContrastiveLoss(0.1, input_channel = self.dim * block.expansion, shape = self._shape_div_2())
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(self.dim * block.expansion, self.num_classes)
        self.fc = nn.Sequential(Flatten(), nn.Linear(int(self.dim * block.expansion * self.shape * self.shape), 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, self.num_classes))

        
    def train_step(self, x , y):
        loss = 0
        
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        loss += self.loss1(output, y)
        output = output.detach()
        
        output = self.layer2(output)
        loss += self.loss2(output, y)
        output = output.detach()
        
        output = self.layer3(output)
        loss += self.loss3(output, y)
        output = output.detach()
        
        output = self.layer4(output)
        loss += self.loss4(output, y)
        output = output.detach()
        
        # output = self.avgpool(output)
        # output = output.view(output.size(0), -1)
        output = self.fc(output)
        loss += self.ce(output, y)
        return loss
    
    def inference(self, x , y):
        output = self.conv1(x)
        # output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        # output = self.avgpool(output)
        # output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class resnet_Research(Resnet_block):
    def __init__(self, args, block, layers):
        super(resnet_Research, self).__init__(args)
        self.num_classes = args.n_classes
        self.block = block
        self.expansion = block.expansion
        self.layers = layers
        self.merge = args.merge
        self.blockwisetotal = args.blockwise_total
        self.ce = nn.CrossEntropyLoss()
        
        self.conv1 = conv_layer_bn(self.num_channel, self.dim, nn.ReLU(inplace=True), stride = 1, kernel_size=3)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer = nn.ModuleList()
        self.loss = nn.ModuleList()
        self.classifier = nn.ModuleList()
        
        self.layer.append(self._make_layer(block = self.block, out_channels = self.dim, blocks = self.layers[0]))
        self.loss.append(Set_Local_Loss(input_channel = self.dim * self.expansion, shape = self.shape,  args = args))
        self.classifier.append(Layer_Classifier(input_channel = (self.dim * self.shape * self.shape * self.expansion), args = args))
        for i in range(1 , self.blockwisetotal):
            self.layer.append(self._make_layer(block = self.block, out_channels = self._dim_mul_2(), stride = 2, blocks = self.layers[i]))
            self.loss.append(Set_Local_Loss(input_channel = self.dim * self.expansion, shape = self._shape_div_2(),  args = args))
            self.classifier.append(Layer_Classifier(input_channel = (self.dim * self.shape * self.shape * self.expansion), args = args))     
        
    def train_step(self, x, y):
        total_loss = 0
        total_classifier_loss = 0
    
        # Layer1
        output = self.conv1(x)
        for i in range(self.blockwisetotal):
            loss, classifier_loss, output = self._training_each_layer(output, y , self.layer[i], self.loss[i], self.classifier[i])
            total_loss += loss
            total_classifier_loss += classifier_loss
             
        return (total_loss + total_classifier_loss) 
    
    def inference(self, x, y):
        classifier_output = {i: [] for i in range(1, self.blockwisetotal + 1)}
    
        # Layer1
        output = self.conv1(x)
        for i in range(self.blockwisetotal):
            classifier_out, output = self._inference_each_layer(output, y , self.layer[i], self.loss[i], self.classifier[i])
            classifier_output[i+1].append(classifier_out)
             
        return output ,  classifier_output

    def _training_each_layer(self, x, y , layer, localloss, classifier, freeze = False):
        classifier_loss = 0
        output = layer(x)
        if freeze:
            output = output.detach()
        loss , projector_out= localloss(output, y)
         
        # projector_out = projector_out.detach()
        if freeze:
            projector_out = projector_out.detach()
        else:
            output = output.detach()
            
        if self.merge == 'merge':
            classifier_out = classifier(projector_out)
        elif self.merge == 'unmerge':
            classifier_out = classifier(output) 
        if freeze:
            classifier_out = classifier_out.detach()
        classifier_loss = self.ce(classifier_out , y) * 0.001
            
        return loss , classifier_loss , output    
    
    def _inference_each_layer(self, x, y , layer, localloss, classifier):
        output = layer(x)
        loss , projector_out= localloss(output, y)
            
        if self.merge == 'merge':
            classifier_out = classifier(projector_out)
        elif self.merge == 'unmerge':
            classifier_out = classifier(output)
                  
        return classifier_out, output

class resnet_Research_Adaptive(resnet_Research):
    def __init__(self, args, block, layers):
        super(resnet_Research_Adaptive, self).__init__(args, block, layers)
        self.countthreshold = args.patiencethreshold
        self.costhreshold = args.cosinesimthreshold
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def inference(self, x, y):
        self.patiencecount = 0
         # Layer1
        output = self.conv1(x)
        classifier_out1, output = self._inference_each_layer(output, y , self.layer1, self.loss1, self.classifier1)
        
        # Layer2    
        classifier_out2, output = self._inference_each_layer(output, y , self.layer2, self.loss2, self.classifier2)
        self.patiencecount += self.AdaptiveCondition(classifier_out1 , classifier_out2)
        if self.patiencecount >= self.countthreshold:
            return classifier_out2
            
        # Layer3
        classifier_out3, output = self._inference_each_layer(output, y , self.layer3, self.loss3, self.classifier3)
        self.patiencecount += self.AdaptiveCondition(classifier_out2 , classifier_out3)
        if self.patiencecount >= self.countthreshold:
            return classifier_out3
            
        # Layer4
        classifier_out4, output = self._inference_each_layer(output, y , self.layer4, self.loss4, self.classifier4)
        return classifier_out4 
    
    def AdaptiveCondition(self, fisrtlayer , prelayer):
        fisrtlayer_maxarg = torch.argmax(fisrtlayer)
        prelayer_maxarg = torch.argmax(prelayer)
        cossimi = torch.mean(self.cos(fisrtlayer , prelayer))
        if fisrtlayer_maxarg == prelayer_maxarg and cossimi > self.costhreshold:
            return  1
        
        return 0    

  
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



