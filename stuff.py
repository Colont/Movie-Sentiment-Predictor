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
def tokenize(text):
    return [token.text for token in nlp(text)]

def preprocess(text):
    text = re.sub(r'<.*?>|/>|\\|[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text.strip())
def tokenize_and_remove_stopwords(text):
    return [token.text for token in nlp(text) if not token.is_stop]

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
    df_train['review'] = df_train['review'].apply(preprocess)
    df_test['review'] = df_test['review'].apply(preprocess)
    '''
    Receive number of tokens
    '''
    df_train['tokenized_review'] = [
        [token.text for token in doc]  
        for doc in nlp.pipe(df_train['review'], batch_size=1000)
    ]

    df_test['tokenized_review'] = [
        [token.text for token in doc]  
        for doc in nlp.pipe(df_test['review'], batch_size=1000)
    ]
    '''
    Remove stop words
    '''

    df_train['tokenized_review'] = [
        [token.text for token in doc if not token.is_stop]  
        for doc in nlp.pipe(df_train['review'], batch_size=1000)
    ]

    df_test['tokenized_review'] = [
        [token.text for token in doc if not token.is_stop]  
        for doc in nlp.pipe(df_test['review'], batch_size=1000)
    ]



    all_tokens = [token for review in df_train['tokenized_review'] for token in review]
    token_counts = Counter(all_tokens)
    vocab = {word: i for i, (word,_) in enumerate(token_counts.most_common(10000), start=1)}
    print(list(vocab.items())[:30])