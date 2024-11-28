import tarfile
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from collections import Counter
import spacy
import os
import torch.optim as optim
import re

#Commented out code that created the parquet file as the runtime is very long, using parquet to made data
#frames that run quick.

#gz_path ='C:/Users/cjbea/Documents/GitHub/main/DS5110/imdb_reviews.gz'
extract_path = 'C:/Users/cjbea/Documents/GitHub/main/DS5110/extraction'
vocab_file = "C:/Users/cjbea/Documents/GitHub/main/DS5110/extraction/aclImdb/imdb.vocab"
vocab = {}
with open(vocab_file, 'r',encoding='latin-1') as file:
    for idx, line in enumerate(file, start=1): 
        word = line.strip()  
        vocab[word] = idx
def preprocess(text):
    text = re.sub(r'<.*?>|/>|\\|[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text.strip())
'''
with tarfile.open(gz_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)
    '''
'''
def load_data(folder, pos_neg):
    data = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as file:
            review = file.read()
            data.append((review, pos_neg))
    return data

'''

nlp = spacy.blank('en')

def tokenize_with_vocab(text, vocab):
    
    # Tokenize and filter in batches for better performance
    tokenized_reviews = []
    for doc in nlp.pipe(text, batch_size=1000, disable=["parser", "ner", "tagger"]):
        tokens = [
            token.text.lower() for token in doc
            if token.text.lower() in vocab and not token.is_stop and not token.is_punct
        ]
        tokenized_reviews.append(tokens)
    return tokenized_reviews
def tokenize_id(tokens, vocab):
    return [vocab[token] for token in tokens]

if __name__ == "__main__":
    '''
    train_pos = load_data(f"{extract_path}/aclImdb/train/pos", 1)
    train_neg = load_data(f"{extract_path}/aclImdb/train/neg", 0)
    test_pos = load_data(f"{extract_path}/aclImdb/test/pos", 1)
    test_neg = load_data(f"{extract_path}/aclImdb/test/neg", 0)
    train_data = pd.DataFrame(train_pos + train_neg, columns=['review', 'label'])
    test_data = pd.DataFrame(test_pos + test_neg, columns=['review', 'label'])
    train_data.to_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/train_data.parquet', engine='pyarrow')
    test_data.to_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/test_data.parquet', engine='pyarrow')
    
    '''
    #convert to paraquet for faster run time
    df_test = pd.read_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/test_data.parquet', engine='pyarrow') 
    df_train = pd.read_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/train_data.parquet', engine='pyarrow')

    #remove useless characters like punctation and html
    df_test['review']= df_test['review'].apply(preprocess)
    df_train['review']=df_train['review'].apply(preprocess)
    
    df_train['tokenized_review'] = tokenize_with_vocab(df_train['review'].values.tolist(), vocab)
    df_test['tokenized_review'] = tokenize_with_vocab(df_test['review'].values.tolist(), vocab)
    #convert tokens to ids using the vocab dictionary
    df_train['review_ids'] = df_train['tokenized_review'].apply(lambda tokens: tokenize_id(tokens, vocab))
    df_test['review_ids'] = df_test['tokenized_review'].apply(lambda tokens: tokenize_id(tokens, vocab))
    #figure out max length for padded sequence 
    df_train['review_length'] = df_train['tokenized_review'].apply(len)
    df_test['review_length'] = df_test['tokenized_review'].apply(len)
    max_length = int(df_train['review_length'].quantile(0.95))

    sequence_train = [torch.tensor(ids, dtype=torch.long) for ids in df_train['review_ids']]
    sequence_test = [torch.tensor(ids, dtype=torch.long) for ids in df_test['review_ids']]
  
    train_padded = rnn_utils.pad_sequence(sequence_train,
                                                     batch_first=True,padding_value=0)
    test_padded  = rnn_utils.pad_sequence(sequence_test,
                                                    batch_first=True,padding_value=0)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        
        
        return self.fc(hidden)
    batch_size = 32
    seq_len = 50
    vocab_size = 1000
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)
input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_data)
print(output.shape)
    #print(train_padded[0])
    #print(test_padded[0])
    
   