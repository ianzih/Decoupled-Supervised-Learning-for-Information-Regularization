import torch
import torch.nn as nn
from utils import conv_layer_bn, Flatten, ALComponent, ContrastiveLoss, PredSimLoss, VICRIG, Layer_Classifier, Set_Local_Loss

class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()
        self.num_class = args.n_classes
        self.in_channel = 3
        self.ce = nn.CrossEntropyLoss()

        self.layer1 = self._make_layer([128, 128, 128, 256, 'M'])
        self.layer2 = self._make_layer([256, 512, 'M'])
        self.layer3 = self._make_layer([512, 512, 'M'])
        self.layer4 = self._make_layer([512, 'M'])

        # self.loss =  ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = 2)
        # self.loss =  VICRIG(c_in = 512, shape = 2, n_class = self.num_class)

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, self.num_class))
        
    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # if self.training:
        #     loss += self.loss(out, y)
        #     out = out.detach()
        
        out = self.fc(out)
        if self.training:
            loss = self.ce(out, y)
            return loss
        else:
            return out.detach()
        
    def _make_linear_layer(self, in_features, out_features):
            return nn.Sequential(nn.Linear(in_features, out_features, bias=True), nn.BatchNorm1d(out_features), nn.ReLU())
        
    def _make_layer(self, channel_size: list):
        layers = []
        for dim in channel_size:
            if dim == 'M':
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channel, dim, nn.ReLU()))
                self.in_channel = dim
        return nn.Sequential(*layers)

class VGG_AL_Component(ALComponent):
    def __init__(self, conv:nn.Module, flatten_size: int, hidden_size: int, out_features: int):

        g_function = nn.Sigmoid() 
        b_function = nn.Sigmoid()

        f = conv
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features), g_function)

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(VGG_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca)

class VGG_AL(nn.Module):
    def __init__(self, args):
        super(VGG_AL, self).__init__()

        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        neuron_size = 500
        self.shape = 32
        self.num_classes = args.n_classes

        layer1 = self._make_layer([128, 256, "M"])
        self.layer1 = VGG_AL_Component(layer1, self.shape*self.shape*256, neuron_size, self.num_classes)

        self.shape //= 2
        layer2 = self._make_layer([256, 512, "M"])
        self.layer2 = VGG_AL_Component(layer2, self.shape*self.shape*512, neuron_size, self.num_classes)

        self.shape //= 2
        layer3 = self._make_layer([512, "M"])
        self.layer3 = VGG_AL_Component(layer3, self.shape*self.shape*512, neuron_size, self.num_classes)

        self.shape //= 2
        layer4 = self._make_layer([512, "M"])
        self.layer4 = VGG_AL_Component(layer4, self.shape*self.shape*512, neuron_size, self.num_classes)

    def forward(self, x, y):
        if self.training:

            y_onehot = torch.zeros([len(y), self.num_classes]).cuda(non_blocking=True)
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
    
    def _make_layer(self, channel_size):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                self.size/=2
            else:
                layers.append(conv_layer_bn(self.features, dim, nn.ReLU()))
                self.features = dim
        return nn.Sequential(*layers)

class VGG_SCPL(nn.Module):
    def __init__(self, args):
        super(VGG_SCPL, self).__init__()
        self.num_classes = args.n_classes

        self.shape = 32
        self.in_channels = 3
        # layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        layer_cfg = {0:[128, 128, 128, 256, "M"], 1:[256, 512, "M"], 2:[512, 512, "M"], 3:[512, "M"]}

        self.layer1 = self._make_layer(layer_cfg[0])
        self.loss1 =  ContrastiveLoss(0.1, input_channel = 256, shape = self._shape_div_2())
        # self.loss1 =  VICRIG(c_in = 256, shape = self._shape_div_2(), mid_neurons = 2048, n_class = self.num_classes)
        
        self.layer2 = self._make_layer(layer_cfg[1])
        self.loss2 =  ContrastiveLoss(0.1, input_channel = 512, shape = self._shape_div_2())
        # self.loss2 =  VICRIG(c_in = 512, shape = self._shape_div_2(), mid_neurons = 2048, n_class = self.num_classes)

        self.layer3 = self._make_layer(layer_cfg[2])
        self.loss3 =  ContrastiveLoss(0.1, input_channel = 512, shape = self._shape_div_2())
        # self.loss3 =  VICRIG(c_in = 512, shape = self._shape_div_2(), mid_neurons = 2048, n_class = self.num_classes)

        self.layer4 = self._make_layer(layer_cfg[3])
        self.loss4 =  ContrastiveLoss(0.1, input_channel = 512, shape = self._shape_div_2())
        # self.loss4 =  VICRIG(c_in = 512, shape = self._shape_div_2(), mid_neurons = 2048, n_class = self.num_classes)

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, self.num_classes))
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, y):
        loss = 0
        output = self.layer1(x)
        if self.training:
            loss += self.loss1(output, y)
            output = output.detach()
        output = self.layer2(output)
        if self.training:
            loss += self.loss2(output, y)
            output = output.detach()
        output = self.layer3(output)
        if self.training:
            loss += self.loss3(output, y)
            output = output.detach()
        output = self.layer4(output)
        if self.training:
            loss += self.loss4(output, y)
            output = output.detach()
        
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss 
        else:
            return output
    
    def _make_layer(self, channel_size):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channels, dim, nn.ReLU()))
                self.in_channels = dim
        return nn.Sequential(*layers)
    
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape


class VGG_SCPL_Dynamic(nn.Module):
    def __init__(self, args):
        super(VGG_SCPL_Dynamic, self).__init__()
        self.num_classes = args.n_classes
        self.merge = args.merge
        self.blockwisetotal = args.blockwise_total
        self.shape = 32
        self.in_channels = 3
        # layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        # layer_cfg = {0: [64, 64, "M" ], 1:[128, 128, "M"], 2:[256, 256, 256, "M"], 3:[512, 512, 512, "M"], 4:[512, 512, 512, "M"]}
        layer_cfg = {0:[128, 128, 128, 256, "M"], 1:[256, 512, "M"], 2:[512, 512, "M"], 3:[512, "M"]}

        self.layer1 = self._make_layer(layer_cfg[0])
        self.loss1 =  Set_Local_Loss(input_channel = self.in_channels, shape = self._shape_div_2(), args = args)
        self.classifier1 = Layer_Classifier(input_channel = (self.in_channels * self.shape * self.shape), args = args)
        
        self.layer2 = self._make_layer(layer_cfg[1])
        self.loss2 =  Set_Local_Loss(input_channel = self.in_channels, shape = self._shape_div_2(), args = args)
        self.classifier2 = Layer_Classifier(input_channel = (self.in_channels * self.shape * self.shape), args = args)
        
        self.layer3 = self._make_layer(layer_cfg[2])
        self.loss3 =  Set_Local_Loss(input_channel = self.in_channels, shape = self._shape_div_2(), args = args)
        self.classifier3 = Layer_Classifier(input_channel = (self.in_channels * self.shape * self.shape), args = args)
        
        self.layer4 = self._make_layer(layer_cfg[3]) 
        self.loss4 =  Set_Local_Loss(input_channel = self.in_channels, shape = self._shape_div_2(), args = args)
        self.classifier4 = Layer_Classifier(input_channel = (self.in_channels * self.shape * self.shape), args = args)
        
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, y):
        total_loss = 0
        total_classifier_loss = 0
        classifier_output = {i: [] for i in range(1, self.blockwisetotal + 1)}
    
        # Layer1
        loss, classifier_loss, classifier_out, output = self._each_layer(x, y , self.layer1, self.loss1, self.classifier1)
        classifier_output[1].append(classifier_out)
        if self.training: 
            total_loss += loss
            total_classifier_loss += classifier_loss
        
        # Layer2    
        loss, classifier_loss, classifier_out, output = self._each_layer(output, y , self.layer2, self.loss2, self.classifier2)
        classifier_output[2].append(classifier_out)
        if self.training: 
            total_loss += loss
            total_classifier_loss += classifier_loss
            
        # Layer3
        loss, classifier_loss, classifier_out, output = self._each_layer(output, y , self.layer3, self.loss3, self.classifier3)
        classifier_output[3].append(classifier_out)
        if self.training: 
            total_loss += loss
            total_classifier_loss += classifier_loss
            
        # Layer4
        loss, classifier_loss, classifier_out, output = self._each_layer(output, y , self.layer4, self.loss4, self.classifier4)
        classifier_output[4].append(classifier_out)
        if self.training: 
            total_loss += loss
            total_classifier_loss += classifier_loss
             
        if self.training:
            return (total_loss + total_classifier_loss) 
        else:
            return output ,  classifier_output
    
    def _make_layer(self, channel_size):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channels, dim, nn.ReLU()))
                self.in_channels = dim
        return nn.Sequential(*layers)
    
    def _shape_div_2(self):
        self.shape //= 2
        return self.shape
    
    def _each_layer(self, x, y , layer, localloss, classifier):
        classifier_loss = 0
        output = layer(x)
        loss , projector_out= localloss(output, y)
        if self.training:
            output = output.detach()
            # projector_out = projector_out.detach()
            
        if self.merge == 'merge':
            classifier_out = classifier(projector_out)
        elif self.merge == 'unmerge':
            classifier_out = classifier(output)
            
        if self.training:
            classifier_loss = self.ce(classifier_out , y) * 0.001
            
            
        return loss , classifier_loss, classifier_out, output
    
    
class VGG_PredSim(nn.Module):
    def __init__(self, args):
        super(VGG_PredSim, self).__init__()
        self.num_classes = args.n_classes

        self.num_layers = 4
        self.size = 32
        self.features = 3
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        conv_layers = []
        loss_funcs = []
        for i in range(self.num_layers):
            conv_layers.append(self._make_conv_layer(layer_cfg[i], nn.ReLU()))
            loss_funcs.append(PredSimLoss(0.1, input_neurons = 2048, c_in = layer_cfg[i][-2], shape = self.size))

        self.flatten = Flatten()

        self.fc = nn.Sequential(Flatten(), nn.Linear(2048, 2500), nn.Sigmoid(), nn.Linear(2500, self.num_classes))
        self.conv = nn.ModuleList(conv_layers)
        self.cl = nn.ModuleList(loss_funcs)
        self.ce = nn.CrossEntropyLoss()

        
    def forward(self, x, label):
        if self.training:
            half = x.shape[0]//2
            x1 = x

            total_loss = 0
            for i in range(self.num_layers):
                x1 = self.conv[i](x1)
                total_loss+=self.cl[i](x1, label)
                x1 = x1.detach()

            y = self.fc(x1[0:half].detach())
            total_loss += self.ce(y, label[0:half])

            return total_loss
        else:
            for i in range(self.num_layers):
                x = self.conv[i](x)
            y = self.fc(x)

            return y
    
    def _make_conv_layer(self, channel_size, activation):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                self.size/=2
            else:
                layers.append(conv_layer_bn(self.features, dim, activation))
                self.features = dim
        return nn.Sequential(*layers)
    def _make_linear_layer(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features, bias=True), nn.BatchNorm1d(out_features), nn.ReLU())