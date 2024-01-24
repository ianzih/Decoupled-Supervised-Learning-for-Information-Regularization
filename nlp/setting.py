from torch.utils.data import DataLoader
from torchtext import datasets

from utils.nlp_utils import *
from utils.utils import *
from nlp.model.LSTM import *
from nlp.model.Transformer import *

import pandas as pd

def set_optim(model, optimal='LARS', args = None):
    if optimal == 'LARS':
        optimizer = LARS(model.parameters(), lr = args.base_lr, weight_decay = args.wd, weight_decay_filter = LARS.exclude_bias_and_norm, 
        lars_adaptation_filter = LARS.exclude_bias_and_norm)
    elif optimal == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.base_lr)
    elif optimal == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.base_lr)
    
    return optimizer
    

def set_loader(dataset, args):
    if dataset == "IMDB":
        from sklearn.model_selection import train_test_split
        
        args.max_len = 350
        n_classes = 2
        df = pd.read_csv('./nlpdataset/IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        
        # corpus = [word for text in df['cleaned_reviews']for word in text.split()]
        corpus = [tokenize(text) for text in df['cleaned_reviews']]
        
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        
        random_seed = 42 if args.side_dim != None else None
        clean_train, clean_test, train_label, test_label = train_test_split(
            text, label, test_size=0.2, random_state = random_seed)
        
        clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
        clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)
        
    elif dataset == "agnews" or dataset == "DBpedia":
        if dataset == "agnews":
            train_data = datasets.AG_NEWS(split='train')
            test_data = datasets.AG_NEWS(split='test')
            args.max_len = 350
            n_classes = 4
        elif dataset == "DBpedia":
            train_data = datasets.DBpedia(split='train')
            test_data = datasets.DBpedia(split='test')
            args.max_len = 60
            n_classes = 14
    
        train_text = [t for _ , t in train_data]
        test_text = [t for _ , t in test_data]
        train_label = [l-1 for l , _ in train_data]
        test_label = [l-1 for l , _ in test_data]
        
        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]
        
        clean_train, train_label = data_cleansing(clean_train, train_label, doRemove=True)
        clean_test, test_label = data_cleansing(clean_test, test_label, doRemove=True)
        
        vocab = create_vocab(clean_train)
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))
        
    
    trainset = Textset(clean_train, train_label, vocab, args.max_len)
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    train_loader = DataLoader(trainset, batch_size = args.train_bsz, collate_fn = trainset.collate, shuffle=True, pin_memory=True)
    test_loader = DataLoader(testset, batch_size = args.test_bsz, collate_fn = testset.collate, pin_memory=True)
    
    if float(args.noise_rate) != 0:
        add_noise(train_loader, n_classes, float(args.noise_rate)) 
    word_vec = get_word_vector(vocab, args.word_vec)
    return train_loader, test_loader, n_classes, word_vec

def set_model(name , args):
    if name == "LSTM":
        model = LSTM(args)
    elif name == "LSTM_Research":
        model = LSTM_Research(args)
    elif name == "LSTM_SCPL":
        model = LSTM_SCPL(args)
    elif name == "LSTM_AL":
        model = LSTM_AL(args)
    elif name == "LSTM_Research_side":
        model = LSTM_Research_side(args)
    elif name == "LSTM_Research_Adaptive":
        model = LSTM_Research_Adaptive(args)
    elif name == "Transformer":
        model = Transformer(args)
    elif name == "Transformer_SCPL":
        model = Transformer_SCPL(args)
    elif name == "Transformer_AL":
        model = Transformer_AL(args)
    elif name == "Transformer_Research":
        model = Transformer_Research(args)
    elif name == "Transformer_Research_side":
        model = Transformer_Research_side(args)
    else:
        raise ValueError("Model not supported: {}".format(name))
    
    return model