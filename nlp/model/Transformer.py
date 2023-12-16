import torch
import torch.nn as nn
from utils.utils import *
from utils.nlp_utils import *
from .transformer.encoder import TransformerEncoder

class Transformer_block(nn.Module):
    def __init__(self, args):
        super(Transformer_block, self).__init__()
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
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), act_fun, nn.Linear(hidden_dim, out_dim))
          
    def _make_layer(self, in_dim, out_dim, word_vec_type = None, n_head = 6):
        if word_vec_type != None:  
            if word_vec_type == "pretrain":
                print("[Layer] Use pretrained embedding")
                layers = nn.Embedding.from_pretrained(self.word_vec, freeze=False)
            else:
                layers = nn.Embedding(in_dim, out_dim)
        else:
            layers = TransformerEncoder(d_model = in_dim, d_ff = out_dim, n_heads = n_head)
        return layers
    
    def sidedata(self, x , dim = -1):
        assert len(x[dim]) == sum(self.side_dim)
        if self.model_weights_path == None:
            return torch.split(x, self.side_dim, dim)[0]
        else:
            return torch.split(x, self.side_dim, dim)

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
    
    def reduction(self, x, mask):
        return torch.sum(x * mask.unsqueeze(-1), dim=1) / torch.sum(mask, -1, keepdim=True)
    
    def forward(self, x, y):
        if self.training:
            return self.train_step(x, y)
        else:
            return self.inference(x, y)
    

class Transformer(Transformer_block):
    def __init__(self, args):
        super(Transformer, self).__init__(args)
        self.ce = nn.CrossEntropyLoss()
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.layer1 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim, n_head = self.n_heads)
        self.layer2 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim, n_head = self.n_heads)
        self.layer3 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim, n_head = self.n_heads)
        self.layer4 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim, n_head = self.n_heads)
        
        self.fc = self.predictlayer(in_dim = self.h_dim, hidden_dim = self.h_dim, out_dim = self.n_classes, act_fun = nn.Tanh())
        
    def train_step(self, x, y):
        mask = self.get_mask(x)
        # embedding
        emb = self.embedding(x)
        # Transformer1
        output = self.layer1(emb, mask)
        # Transformer2
        output = self.layer2(output, mask)
        # Transformer3
        output = self.layer3(output, mask)
        # Transformer4
        output = self.layer4(output, mask)
        
        output = self.fc(self.reduction(output, mask))
        loss = self.ce(output, y)
        
        return loss

    def inference(self, x, y):
        mask = self.get_mask(x)
        # embedding
        emb = self.embedding(x)
        # Transformer1
        output = self.layer1(emb, mask)
        # Transformer2
        output = self.layer2(output, mask)
        # Transformer3
        output = self.layer3(output, mask)
        # Transformer4
        output = self.layer4(output, mask)
        
        output = self.fc(self.reduction(output, mask))
        
        return output


class Transformer_SCPL(Transformer_block):
    def __init__(self, args):
        super(Transformer_SCPL, self).__init__(args)
        # embedding
        self.embedding = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.loss0 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer1
        self.layer1 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim)
        self.loss1 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer2
        self.layer2 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim)
        self.loss2 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer3
        self.layer3 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim)
        self.loss3 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        # layer4
        self.layer4 = self._make_layer(in_dim = self.h_dim, out_dim = self.h_dim)
        self.loss4 =  ContrastiveLoss(0.1, input_channel = self.h_dim, shape = 1, mid_neurons = self.h_dim, out_neurons = self.h_dim, args = args)
        
        self.fc = self.predictlayer(in_dim = self.h_dim, hidden_dim = self.h_dim, out_dim = self.n_classes, act_fun = nn.Tanh())
        self.ce = nn.CrossEntropyLoss()
        
    def train_step(self, x, y):
        loss = 0
        mask = self.get_mask(x)
        # embedding
        emb = self.embedding(x)
        loss += self.loss0(emb.mean(1), y)
        emb = emb.detach()
        # Transformer1
        output = self.layer1(emb, mask)
        loss += self.loss1(self.reduction(output, mask), y)
        # Transformer2
        output = self.layer2(output.detach(), mask)
        loss += self.loss2(self.reduction(output, mask), y)
        # Transformer3
        output = self.layer3(output.detach(), mask)
        loss += self.loss3(self.reduction(output, mask), y)
        # Transformer4
        output = self.layer4(output.detach(), mask)
        loss += self.loss4(self.reduction(output, mask), y)
        
        output = output.detach()
        output = self.fc(self.reduction(output, mask))
        loss += self.ce(output, y)
        return loss
          
    def inference(self, x, y):
        mask = self.get_mask(x)
        # embedding
        emb = self.embedding(x)
        # Transformer1
        output = self.layer1(emb, mask)
        # Transformer2
        output = self.layer2(output, mask)
        # Transformer3
        output = self.layer3(output, mask)
        # Transformer4
        output = self.layer4(output, mask)
        
        output = self.fc(self.reduction(output, mask))
         
        return output


class Transformer_Research(Transformer_block):
    def __init__(self, args):
        super(Transformer_Research,self).__init__(args)
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
        hidden = None
        total_loss = 0
        total_classifier_loss = 0
        if self.side_dim != None and self.modeltype == "Transformer_Research":
            x = self.sidedata(x)
        
        emb = self.embedding(x)
        loss , _= self.lossemb(emb.mean(1), y)
        emb = emb.detach()
        total_loss += loss
        
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            loss , classifier_loss , output , hidden = self._training_each_layer(output, y, hidden, self.layer[i], self.loss[i], self.classifier[i])
            total_loss += loss
            total_classifier_loss += classifier_loss
            
        if self.modeltype == "Transformer_Research_side":
            return (total_classifier_loss + total_loss) , output , hidden
        else:
            return (total_classifier_loss + total_loss)
    
    def inference(self, x, y):
        hidden = None
        classifier_output = {i: [] for i in range(1, self.blockwisetotal)}
        
        emb = self.embedding(x)
        output = emb
        for i in range(0 , self.blockwisetotal - 1):
            classifier_out, output , hidden = self._inference_each_layer(output, y, hidden, self.layer[i], self.loss[i], self.classifier[i])
            classifier_output[i+1].append(classifier_out)
            
        if self.modeltype == "Transformer_Research_side":
            return output ,  classifier_output , hidden
        else:
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
    
    
class Transformer_Research_side(Transformer_Research):
    def __init__(self, args):
        super(Transformer_Research_side , self).__init__(args)
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
        _ , output , hidden= super(Transformer_Research_side, self).train_step(x, y)
        
        # Input side data to new layer
        output = torch.cat((output[:, :, :self.h_dim], output[:, :, self.h_dim:]), dim = 1)
        x_cat = torch.cat((output, emb_side), dim = 1)
        loss , classifier_loss , output , hidden = self._training_each_layer(x_cat, y, hidden, self.newlayer, self.newloss, self.newclassifier)
        total_loss += loss
        total_classifier_loss += classifier_loss
        
        return (total_classifier_loss + total_loss)
    
    def inference(self, x, y):
        classifier_output = {i: [] for i in range(1, self.blockwisetotal + 1)}
        
        x , x_side = self.sidedata(x)
        emb_side = self.embedding(x_side)
        output ,  classifier_output_pre , hidden = super(Transformer_Research_side, self).inference(x, y)
        for key, value in classifier_output_pre.items():
            classifier_output[key] = value
        
        # Input side data to new layer
        output = torch.cat((output[:, :, :self.h_dim], output[:, :, self.h_dim:]), dim = 1)
        x_cat = torch.cat((output, emb_side), dim = 1)
        classifier_out, output , hidden = self._inference_each_layer(x_cat, y, hidden, self.newlayer, self.newloss, self.newclassifier)
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
        for param in super(Transformer_Research_side, self).parameters():
            param.requires_grad = False