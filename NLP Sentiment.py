import tarfile
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
import os
import torch.optim as optim
import re
from torch.utils.data import Dataset, DataLoader

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

    df_test = pd.read_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/test_data.parquet', engine='pyarrow') 
    df_train = pd.read_parquet('C:/Users/cjbea/Documents/GitHub/main/DS5110/train_data.parquet', engine='pyarrow')
    df_test['review']= df_test['review'].apply(preprocess)
    df_train['review']=df_train['review'].apply(preprocess)
    df_train['tokenized_review'] = tokenize_with_vocab(df_train['review'].values.tolist(), vocab)
    df_test['tokenized_review'] = tokenize_with_vocab(df_test['review'].values.tolist(), vocab)

    print(df_train.columns)