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
        self.layer1 = self._make_layer(in_dim = self.vocab_size, out_dim = self.emb_dim, word_vec_type = args.word_vec_type)
        self.layer2 = self._make_layer(in_dim = self.emb_dim, out_dim = self.h_dim)
        self.layer3 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)
        self.layer4 = self._make_layer(in_dim = self.h_dim * 2, out_dim = self.h_dim)

        self.fc = self.predictlayer(in_dim = self.h_dim, hidden_dim = self.h_dim, out_dim = self.n_classes, act_fun = nn.Tanh())
        
    def train_step(self, x, y):
        hidden = None
        # embedding
        emb = self.layer1(x)
        # LSTM1
        out, (h, c) = self.layer2(emb, hidden)
        hidden = (h, c)
        # LSTM2
        out, (h, c) = self.layer3(out, hidden)
        hidden = (h, c)
        # LSTM3
        out, (h, c) = self.layer4(out, hidden)
        
        out = self.fc((h[0] + h[1]) / 2)
        loss = self.ce(out, y)
        return loss

    def inference(self, x, y):
        hidden = None
        # embedding
        emb = self.layer1(x)
        # LSTM1
        out, (h, c) = self.layer2(emb, hidden)
        hidden = (h, c)
        # LSTM2
        out, (h, c) = self.layer3(out, hidden)
        hidden = (h, c)
        # LSTM3
        out, (h, c) = self.layer4(out, hidden)
        
        out = self.fc((h[0] + h[1]) / 2)
         
        return out
