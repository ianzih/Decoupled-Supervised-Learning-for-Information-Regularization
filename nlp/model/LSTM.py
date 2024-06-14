import torch
import torch.nn as nn
from utils.utils import *
from utils.nlp_utils import *


class LSTM_block(nn.Module):
    def __init__(self, args):
        super(LSTM_block, self).__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.emb_dim = args.emb_dim
        self.h_dim = args.h_dim
        self.n_classes = args.n_classes
        self.n_heads = args.heads
        self.word_vec = args.word_vec
        self.blockwisetotal = args.blockwise_total
        self.merge = args.merge
        self.modeltype = args.model
        self.side_dim = args.side_dim
        self.model_weights_path = args.model_weights_path
        
    def predictlayer(self, in_dim, out_dim, hidden_dim=100, act_fun = nn.Tanh()):
        return nn.Sequential(nn.Identity(), act_fun, nn.Linear(in_dim, out_dim))
          
    def _make_layer(self, in_dim, out_dim, word_vec_type = None):
        if word_vec_type != None:  
            if word_vec_type == "pretrain":
                print("[Layer] Use pretrained embedding")
                layers = nn.Embedding.from_pretrained(self.word_vec, freeze=False)
            else:
                layers = nn.Embedding(in_dim, out_dim)
        else:
            layers = nn.LSTM(in_dim, out_dim, bidirectional = True, batch_first = True)
        return layers
    
    def sidedata(self, x , dim = -1):
        assert len(x[dim]) == sum(self.side_dim)
        if self.model_weights_path == None:
            return torch.split(x, self.side_dim, dim)[0]
        else:
            return torch.split(x, self.side_dim, dim)
    
    def forward(self, x, y):
        self.hiddenlist = [None for _ in range(self.blockwisetotal-1)]
        if self.training:
            return self.train_step(x, y)
        else:
            return self.inference(x, y)
    

class LSTM(LSTM_block):
    def __init__(self, args):
        super(LSTM, self).__init__(args)
        self.ce = nn.CrossEntropyLoss()
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.layer1 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim)
        self.layer2 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer3 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer4 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)

        self.fc = self.predictlayer(in_dim = self.h_dim, hidden_dim = self.h_dim, out_dim = self.n_classes, act_fun = nn.Tanh())
        
    def train_step(self, x, y):
        # embedding
        emb = self.embedding(x)
        # LSTM1
        output, self.hiddenlist[0] = self.layer1(emb, self.hiddenlist[0])
        # LSTM2
        output, self.hiddenlist[1] = self.layer2(output, self.hiddenlist[1])
        # LSTM3
        output, self.hiddenlist[2] = self.layer3(output, self.hiddenlist[2])
        # LSTM4
        output, self.hiddenlist[3] = self.layer4(output, self.hiddenlist[3])
        
        output = self.fc((self.hiddenlist[3][0][0] + self.hiddenlist[3][0][1]) / 2)
        loss = self.ce(output, y)
        return loss

    def inference(self, x, y):
        # embedding
        emb = self.embedding(x)
        # LSTM1
        output, self.hiddenlist[0] = self.layer1(emb, self.hiddenlist[0])
        # LSTM2
        output, self.hiddenlist[1] = self.layer2(output, self.hiddenlist[1])
        # LSTM3
        output, self.hiddenlist[2] = self.layer3(output, self.hiddenlist[2])
        # LSTM4
        output, self.hiddenlist[3] = self.layer4(output, self.hiddenlist[3])
        
        output = self.fc((self.hiddenlist[3][0][0] + self.hiddenlist[3][0][1]) / 2)
         
        return output


class LSTM_AL_Component(ALComponent):
    def __init__(self, conv:nn.Module, input_size: int, shape: int, hidden_size: int, out_features: int, catype = None): 
        flatten_size = int(input_size * shape * shape)
        g_function = nn.Tanh() 
        b_function = nn.Tanh()

        f = conv
        g = nn.Sequential(nn.Linear(out_features , hidden_size), g_function)
        b = nn.Sequential(nn.Linear(flatten_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size , out_features), g_function)

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(LSTM_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca, catype)
    
    def forward(self, x = None, y = None, label = None, hidden = None):
        if self.training:
            if self.catype == "emb":
                s = self.f(x)
                s1 = s.mean(1)
            else:
                s, hidden = self.f(x, hidden)
                s1 = (hidden[0][0] + hidden[0][1]) / 2
                hidden = (hidden[0].detach() , hidden[1].detach())
            loss_f = 0
            s0 = self.b(s1)
            t = self.g(y)
            t0 = self.inv(t)
            
            loss_b = self.cb(s0, t.detach()) # local loss
            loss_ae = self.ca(t0, y)
            return s.detach(), t.detach(), loss_f, loss_b, loss_ae, hidden
        else:
            if y == None:
                if self.catype == "emb":
                    s = self.f(x)
                else:
                    s , hidden = self.f(x , hidden)
                return s , hidden
            else:
                t0 = self.inv(y)
                return t0
            
    def bridge_forward(self, x, hidden):
        s, hidden = self.f(x, hidden)
        s1 = (hidden[0][0] + hidden[0][1]) / 2
        s0 = self.b(s1)
        t0 = self.inv(s0)
        return t0


class LSTM_AL(LSTM_block):
    def __init__(self, args):
        super(LSTM_AL, self).__init__(args)
        neuron_size = 300
        self.num_classes = args.n_classes
        embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.embedding = LSTM_AL_Component(conv = embedding, input_size = self.h_dim, shape = 1, hidden_size = neuron_size, out_features = self.num_classes, catype = "emb")
         
        layer1 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim)
        self.layer1 = LSTM_AL_Component(conv = layer1, input_size = self.h_dim, shape = 1, hidden_size = neuron_size, out_features = neuron_size)

        layer2 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer2 = LSTM_AL_Component(conv = layer2, input_size = self.h_dim, shape = 1, hidden_size = neuron_size, out_features = neuron_size)

        layer3 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer3 = LSTM_AL_Component(conv = layer3, input_size = self.h_dim, shape = 1, hidden_size = neuron_size, out_features = neuron_size)

        layer4 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer4 = LSTM_AL_Component(conv = layer4, input_size = self.h_dim, shape = 1, hidden_size = neuron_size, out_features = neuron_size)
    
    def train_step(self, x, y):
        total_loss = 0
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes).float().to(y.device)

        _s = x
        _t = y_onehot
        
        _s, _t, loss_f, loss_b, loss_ae, _ = self.embedding(x = _s , y = _t)
        total_loss += (loss_f + loss_b + loss_ae)

        _s, _t, loss_f, loss_b, loss_ae, self.hiddenlist[0] = self.layer1(_s, _t, y, self.hiddenlist[0])
        total_loss += (loss_f + loss_b + loss_ae)

        _s, _t, loss_f, loss_b, loss_ae, self.hiddenlist[1] = self.layer2(_s, _t, y, self.hiddenlist[1])
        total_loss += (loss_f + loss_b + loss_ae)

        _s, _t, loss_f, loss_b, loss_ae, self.hiddenlist[2] = self.layer3(_s, _t, y, self.hiddenlist[2])
        total_loss += (loss_f + loss_b + loss_ae)

        _s, _t, loss_f, loss_b, loss_ae, self.hiddenlist[3] = self.layer4(_s, _t, y, self.hiddenlist[3])
        total_loss += (loss_f + loss_b + loss_ae)
        return total_loss
    
    def inference(self, x, y):
        _s = x
        _s, _ = self.embedding(_s)
        _s, self.hiddenlist[0] = self.layer1(_s, None, self.hiddenlist[0])
        _s, self.hiddenlist[1] = self.layer2(_s, None, self.hiddenlist[1])
        _s, self.hiddenlist[2] = self.layer3(_s, None, self.hiddenlist[2])
        _t0 = self.layer4.bridge_forward(_s, self.hiddenlist[3])
        _t0 = self.layer3(x = None, y =_t0)
        _t0 = self.layer2(x = None, y =_t0)
        _t0 = self.layer1(x = None, y =_t0)
        _t0 = self.embedding(x = None, y =_t0)
        return _t0
          

class LSTM_SCPL(LSTM_block):
    def __init__(self, args):
        super(LSTM_SCPL, self).__init__(args)
        # embedding
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.loss0 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer1
        self.layer1 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim)
        self.loss1 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer2
        self.layer2 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.loss2 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer3
        self.layer3 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.loss3 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        #layer4
        self.layer4 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.loss4 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        
        self.fc = self.predictlayer(in_dim = self.h_dim, hidden_dim = self.h_dim, out_dim = self.n_classes, act_fun = nn.Tanh())
        self.ce = nn.CrossEntropyLoss()
        
    def train_step(self, x, y):
        loss = 0
        # embedding
        emb = self.embedding(x)
        loss += self.loss0(emb.mean(1), y)
        emb = emb.detach()
        # LSTM1
        output, self.hiddenlist[0] = self.layer1(emb, self.hiddenlist[0])
        loss += self.loss1((self.hiddenlist[0][0][0] + self.hiddenlist[0][0][1]) / 2, y)
        self.hiddenlist[0] = (self.hiddenlist[0][0].detach() , self.hiddenlist[0][1].detach())
        # LSTM2
        output, self.hiddenlist[1] = self.layer2(output.detach(), self.hiddenlist[1])
        loss += self.loss2((self.hiddenlist[1][0][0] + self.hiddenlist[1][0][1]) / 2, y)
        self.hiddenlist[1] = (self.hiddenlist[1][0].detach() , self.hiddenlist[1][1].detach())
        # LSTM3
        output, self.hiddenlist[2] = self.layer3(output.detach(), self.hiddenlist[2])
        loss += self.loss3((self.hiddenlist[2][0][0] + self.hiddenlist[2][0][1]) / 2, y)
        self.hiddenlist[2] = (self.hiddenlist[2][0].detach() , self.hiddenlist[2][1].detach())
        # LSTM4
        output, self.hiddenlist[3] = self.layer4(output.detach(), self.hiddenlist[3])
        loss += self.loss4((self.hiddenlist[3][0][0] + self.hiddenlist[3][0][1]) / 2, y)
        self.hiddenlist[3] = (self.hiddenlist[3][0].detach() , self.hiddenlist[3][1].detach())
        
        output = self.fc(((self.hiddenlist[3][0][0] + self.hiddenlist[3][0][1]) / 2))
        loss += self.ce(output, y)
        return loss
          
    def inference(self, x, y):
        # embedding
        emb = self.embedding(x)
        # LSTM1
        output, self.hiddenlist[0] = self.layer1(emb, self.hiddenlist[0])
        # LSTM2
        output, self.hiddenlist[1] = self.layer2(output, self.hiddenlist[1])
        # LSTM3
        output, self.hiddenlist[2] = self.layer3(output, self.hiddenlist[2])
        # LSTM4
        output, self.hiddenlist[3] = self.layer4(output, self.hiddenlist[3])
        
        output = self.fc(((self.hiddenlist[3][0][0] + self.hiddenlist[3][0][1]) / 2))
         
        return output


class LSTM_DeInfoReg(LSTM_block):
    def __init__(self, args):
        super(LSTM_DeInfoReg,self).__init__(args)
        self.layer = nn.ModuleList()
        self.loss = nn.ModuleList()
        self.classifier = nn.ModuleList()
        
        self.ce = nn.CrossEntropyLoss()
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.lossemb = (Set_Local_Loss(input_channel = self.emb_dim, shape = 1, args = args, activation = nn.Tanh()))  
        for i in range(0, self.blockwisetotal - 1):
            if i == 0:
                self.layer.append(self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim))
                self.loss.append(Set_Local_Loss(input_channel = self.h_dim, shape = 1, args = args, activation = nn.Tanh()))
                self.classifier.append(Layer_Classifier(input_channel = self.h_dim, args = args, activation = nn.Tanh()))
            else:
                self.layer.append(self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim))
                self.loss.append(Set_Local_Loss(input_channel = self.h_dim, shape = 1, args = args, activation = nn.Tanh()))
                self.classifier.append(Layer_Classifier(input_channel = self.h_dim, args = args, activation = nn.Tanh()))
        
    def train_step(self, x, y):
        total_loss = 0
        total_classifier_loss = 0
        if self.side_dim != None and self.modeltype == "LSTM_DeInfoReg":
            x = self.sidedata(x)
        
        emb = self.embedding(x)
        loss , _= self.lossemb(emb.mean(1), y)
        emb = emb.detach()
        total_loss += loss
        
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            loss , classifier_loss , output , self.hiddenlist[i] = self._training_each_layer(output, y, self.hiddenlist[i], self.layer[i], self.loss[i], self.classifier[i])
            total_loss += loss
            total_classifier_loss += classifier_loss
            
        if self.modeltype == "LSTM_DeInfoReg_side":
            return (total_classifier_loss + total_loss) , output
        else:
            return (total_classifier_loss + total_loss)
    
    def inference(self, x, y):
        classifier_output = {i: [] for i in range(1, self.blockwisetotal)}
        
        emb = self.embedding(x)
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            classifier_out, output , self.hiddenlist[i] = self._inference_each_layer(output, y, self.hiddenlist[i], self.layer[i], self.loss[i], self.classifier[i])
            classifier_output[i+1].append(classifier_out)
            
        return output ,  classifier_output
         
    def _training_each_layer(self, x, y, hidden, layer, localloss, classifier, freeze = False):
        output, hidden = layer(x , hidden)
        if freeze:
            output = output.detach()
        loss , projector_out= localloss((hidden[0][0] + hidden[0][1]) / 2, y)
         
        if freeze:
            projector_out = projector_out.detach()
        else:
            output = output.detach()
            hidden = (hidden[0].detach() , hidden[1].detach())
            
        if self.merge == 'merge':
            classifier_out = classifier(projector_out)
        elif self.merge == 'unmerge':
            classifier_out = classifier((hidden[0][0] + hidden[0][1]) / 2) 
        if freeze:
            classifier_out = classifier_out.detach()
        classifier_loss = self.ce(classifier_out , y) * 0.001
            
        return loss , classifier_loss , output, hidden
    
    def _inference_each_layer(self, x, y, hidden, layer, localloss, classifier):
        output, hidden = layer(x , hidden)
        _ , projector_out= localloss((hidden[0][0] + hidden[0][1]) / 2, y)
            
        if self.merge == 'merge':
            classifier_out = classifier(projector_out)
        elif self.merge == 'unmerge':
            classifier_out = classifier((hidden[0][0] + hidden[0][1]) / 2) 
                  
        return classifier_out, output, hidden
    
    
class LSTM_DeInfoReg_side(LSTM_DeInfoReg):
    def __init__(self, args):
        super(LSTM_DeInfoReg_side , self).__init__(args)
        self.side_dim = args.side_dim
        args.blockwise_total += 1
        
        # Need to use pretrain model
        if args.model_weights_path != None:
            self.load_model_weights(args.model_weights_path)    
            self.freeze_pretrain_model()
            
        self.newlayer = self._make_layer(in_dim = self.h_dim , out_dim = self.h_dim)
        self.newloss = Set_Local_Loss(input_channel = self.h_dim, shape = 1, args = args, activation = nn.Tanh())
        self.newclassifier = Layer_Classifier(input_channel = self.h_dim, args = args, activation = nn.Tanh())
        
    def train_step(self, x, y):
        total_loss = 0
        total_classifier_loss = 0
        
        x , x_side = self.sidedata(x)
        emb_side = self.embedding(x_side)
        _ , output = super(LSTM_DeInfoReg_side, self).train_step(x, y)
        
        # Input side data to new layer
        output = torch.cat((output[:, :, :self.h_dim], output[:, :, self.h_dim:]), dim = 1)
        x_cat = torch.cat((output, emb_side), dim = 1)
        loss , classifier_loss , output , self.hiddenlist[-1] = self._training_each_layer(x_cat, y, self.hiddenlist[-1], self.newlayer, self.newloss, self.newclassifier)
        total_loss += loss
        total_classifier_loss += classifier_loss
        
        return (total_classifier_loss + total_loss)
    
    def inference(self, x, y):
        classifier_output = {i: [] for i in range(1, self.blockwisetotal + 1)}
        
        x , x_side = self.sidedata(x)
        emb_side = self.embedding(x_side)
        output ,  classifier_output_pre = super(LSTM_DeInfoReg_side, self).inference(x, y)
        for key, value in classifier_output_pre.items():
            classifier_output[key] = value
        
        # Input side data to new layer
        output = torch.cat((output[:, :, :self.h_dim], output[:, :, self.h_dim:]), dim = 1)
        x_cat = torch.cat((output, emb_side), dim = 1)
        classifier_out, output , self.hiddenlist[-1] = self._inference_each_layer(x_cat, y, self.hiddenlist[-1], self.newlayer, self.newloss, self.newclassifier)
        classifier_output[self.blockwisetotal].append(classifier_out)
        
        return output ,  classifier_output
    
    def load_model_weights(self, model_weights_path):
        try:
            model_state_dict = torch.load(model_weights_path)
            pretrained_dict = model_state_dict["model"] if "model" in model_state_dict else model_state_dict
            # for layer, weights in pretrained_dict.items():
            #     print(f'Layer: {layer}, Shape: {weights.shape}')
            self.load_state_dict(pretrained_dict, strict=False)   
        except Exception as E:
            print("Model Weights Path Does Not Exit")
            os._exit(0)
    
    def freeze_pretrain_model(self):
        # Freeze weights of the pretrain model
        for param in super(LSTM_DeInfoReg_side, self).parameters():
            param.requires_grad = False
            
            
class LSTM_DeInfoReg_Adaptive(LSTM_DeInfoReg):
    def __init__(self, args):
        super(LSTM_DeInfoReg_Adaptive, self).__init__(args)
        self.countthreshold = args.patiencethreshold
        self.costhreshold = args.cosinesimthreshold
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def inference(self, x, y):
        self.patiencecount = 0
        classifier_out_pre = None
        
        emb = self.embedding(x)
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            if i == 0 :
                classifier_out, output , self.hiddenlist[i] = self._inference_each_layer(output, y, self.hiddenlist[i], self.layer[0], self.loss[0], self.classifier[0])
                classifier_out_pre = classifier_out
            else:
                classifier_out, output , self.hiddenlist[i] = self._inference_each_layer(output, y, self.hiddenlist[i], self.layer[i], self.loss[i], self.classifier[i])
                self.patiencecount += self.AdaptiveCondition(classifier_out_pre , classifier_out)
                classifier_out_pre = classifier_out
                 
                if i == self.blockwisetotal - 2:
                    return classifier_out , i
                elif self.patiencecount >= self.countthreshold:
                    return classifier_out , i
    
    def AdaptiveCondition(self, fisrtlayer , prelayer):
        fisrtlayer_maxarg = torch.argmax(fisrtlayer)
        prelayer_maxarg = torch.argmax(prelayer)
        cossimi = torch.mean(self.cos(fisrtlayer , prelayer))
        if fisrtlayer_maxarg == prelayer_maxarg and cossimi > self.costhreshold:
            return  1
        
        return 0  