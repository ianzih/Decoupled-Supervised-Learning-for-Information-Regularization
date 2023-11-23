import torch
import torch.nn as nn
from utils.utils import *
from utils.vision_utils import *


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
        
    def predictlayer(self, in_dim, out_dim, hidden_dim=100, act_fun = nn.Tanh()):
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), act_fun, nn.Linear(hidden_dim, out_dim))
          
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
    
    def forward(self, x, y):
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
        hidden = None
        # embedding
        emb = self.embedding(x)
        # LSTM1
        output, hidden = self.layer1(emb, hidden)
        # LSTM2
        output, hidden = self.layer2(output, hidden)
        # LSTM3
        output, hidden = self.layer3(output, hidden)
        # LSTM4
        output, hidden = self.layer4(output, hidden)
        
        output = self.fc((hidden[0][0] + hidden[0][1]) / 2)
        loss = self.ce(output, y)
        return loss

    def inference(self, x, y):
        hidden = None
        # embedding
        emb = self.embedding(x)
        # LSTM1
        output, hidden = self.layer1(emb, hidden)
        # LSTM2
        output, hidden = self.layer2(output, hidden)
        # LSTM3
        output, hidden = self.layer3(output, hidden)
        # LSTM4
        output, hidden = self.layer4(output, hidden)
        
        output = self.fc((hidden[0][0] + hidden[0][1]) / 2)
         
        return output

class LSTM_Research(LSTM_block):
    def __init__(self, args):
        super(LSTM_Research,self).__init__(args)
        self.layer = nn.ModuleList()
        self.loss = nn.ModuleList()
        self.classifier = nn.ModuleList()
        
        self.ce = nn.CrossEntropyLoss()
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
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
        hidden = None
        total_loss = 0
        total_classifier_loss = 0
        
        emb = self.embedding(x)
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            loss , classifier_loss , output , hidden = self._training_each_layer(output, y , hidden, self.layer[i], self.loss[i], self.classifier[i])
            total_loss += loss
            total_classifier_loss += classifier_loss
        
        return (total_classifier_loss + total_loss)
    
    def inference(self, x, y):
        hidden = None
        classifier_output = {i: [] for i in range(1, self.blockwisetotal)}
        
        emb = self.embedding(x)
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            classifier_out, output , hidden = self._inference_each_layer(output, y , hidden, self.layer[i], self.loss[i], self.classifier[i])
            classifier_output[i+1].append(classifier_out)
        
        return output ,  classifier_output
         
    def _training_each_layer(self, x, y, hidden, layer, localloss, classifier, freeze = False):
        output, hidden = layer(x , hidden)
        if freeze:
            output = output.detach()
        loss , projector_out= localloss((hidden[0][0] + hidden[0][1]) / 2, y)
         
        # projector_out = projector_out.detach()
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