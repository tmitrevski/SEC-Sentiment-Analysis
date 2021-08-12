# %%
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import spacy

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
# %%
filing_path = Path('./Resources/Cleaned Filings/')
for i, filing in enumerate(filing_path.glob('*.txt'), 1):
    if i % 500 == 0:
        print(i, end=' ', flush=True)
    filing_id = int(filing.stem)
    items = {}
    for section in filing.read_text().lower().split('°'):
        if section.startswith('item '):
            if len(section.split()) > 1:
                item = section.split()[1].replace('.', '').replace(':', '').replace(',', '')
                text = ' '.join([t for t in section.split()[2:]])
                if items.get(item) is None or len(items.get(item)) < len(text):
                    items[item] = text

    txt = pd.Series(items).reset_index()
    txt.columns = ['item', 'text']
    sections_path = Path('./Resources/Sections/' + filing.stem + '.csv')
    txt.to_csv(sections_path, index=False)
# %%
csv_path = Path("Resources/10K_10Q_List.csv")
df = pd.read_csv(csv_path)
reqd_Index = df[df['Form Type'] == '10-K'].index.tolist()

sections = ['1', '1a', '7', '7a']
sections_path = Path('./Resources/Sections/')
clean_path = Path('./Resources/Parsed Sections/')
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.max_length = 6000000

vocab = Counter()
t = total_tokens = 0
stats = []
done = 0
for text_file in sections_path.glob('*.csv'):
    file_id = int(text_file.stem)
    if (file_id in reqd_Index):
        clean_file = Path('./Resources/Parsed Sections/' + str(file_id) + '.csv')
        if clean_file.exists():
            continue
        items = pd.read_csv(text_file).dropna()
        items.item = items.item.astype(str)
        items = items[items.item.isin(sections)]
    
        clean_doc = []
        for _, (item, text) in items.iterrows():
            doc = nlp(text)
            for s, sentence in enumerate(doc.sents):
                clean_sentence = []
                if sentence is not None:
                    for t, token in enumerate(sentence, 1):
                        if not any([token.is_stop,
                                    token.is_digit,
                                    not token.is_alpha,
                                    token.is_punct,
                                    token.is_space,
                                    token.lemma_ == '-PRON-',
                                    token.pos_ in ['PUNCT', 'SYM', 'X']]):
                            clean_sentence.append(token.text.lower())
                    total_tokens += t
                    if len(clean_sentence) > 0:
                        clean_doc.append([item, s, ' '.join(clean_sentence)])
        (pd.DataFrame(clean_doc,
                     columns=['item', 'sentence', 'text'])
        .dropna()
        .to_csv(clean_file, index=False))
        done += 1
    if done % 500 == 0:
        print(done, end=' ', flush=True)
# %%
min_sentence_length = 5
max_sentence_length = 50
i = 0
sent_length = Counter()

for text_file in clean_path.glob('*.csv'):
    file_id = int(text_file.stem)
    ngrams1_file = Path('./Resources/ngrams1/' + str(file_id) + '.txt')
    if i % 100 == 0:
        print(i, end=' ', flush=True)
    text = pd.read_csv(text_file).text
    sent_length.update(text.str.split().str.len().tolist())
    text = text[text.str.split().str.len().between(min_sentence_length, max_sentence_length)]
    text = '\n'.join(text.tolist())
    with (ngrams1_file).open('w') as f:
        f.write(text)
    i += 1
# %%
import os
ngrams1_path = Path('./Resources/ngrams1/')
i = 0
for file in ngrams1_path.glob('*.txt'):
    if os.path.getsize(file) == 0 :
        os.remove(file)
# %%

files = ngrams1_path.glob('*.txt')
texts = [f.read_text() for f in files]
unigrams = Path('./Resources/ngrams/ngrams_1.txt')
unigrams.write_text('\n'.join(texts))
texts = unigrams.read_text()

# %%
from babel.dates import format_time
n_grams = []
start = time()
print(time())
for i, n in enumerate([2, 3]):
    sentences = LineSentence(Path('./Resources/ngrams/ngrams_' + str(n-1) +'.txt'))
    phrases = Phrases(sentences=sentences,
                      min_count=25,  # ignore terms with a lower count
                      threshold=0.5,  # accept phrases with higher score
                      max_vocab_size=4000000,  # prune of less common words to limit memory use
                      delimiter=b'_',  # how to join ngram tokens
                      scoring='npmi')

    s = pd.DataFrame([[k.decode('utf-8'), v] for k, v in phrases.export_phrases(sentences)], 
                     columns=['phrase', 'score']).assign(length=n)

    n_grams.append(s.groupby('phrase').score.agg(['mean', 'size']))
    print(n_grams[-1].nlargest(5, columns='size'))
    if n == 3:
        grams = Phraser(phrases)
        sentences = grams[sentences]
        Path('./Resources/ngrams/ngrams_' + str(n) +'.txt').write_text('\n'.join([' '.join(s) for s in sentences]))
        
        src_dir = Path('./Resources/ngrams' + str(n-1) +'/')
        target_dir = Path('./Resources/ngrams' + str(n) +'/')
        if not target_dir.exists():
            target_dir.mkdir()
        
        for f in src_dir.glob('*.txt'):
            text = LineSentence(f)
            text = grams[text]
            (target_dir / f'{f.stem}.txt').write_text('\n'.join([' '.join(s) for s in text]))
    print(time())

n_grams = pd.concat(n_grams).sort_values('size', ascending=False)          
n_grams.to_parquet(Path('./Resources/ngrams.parquet'), compression='UNCOMPRESSED')
# %%
#n_grams = pd.concat(n_grams).sort_values('size', ascending=False)          
n_grams.to_parquet(Path('./Resources/ngrams.parquet'), compression='UNCOMPRESSED')
# %%
n_grams.groupby(n_grams.index.str.replace('_', ' ').str.count(' ')).size()
sentences = (Path('./Resources/ngrams/ngrams_3.txt')).read_text().split('\n')
n = len(sentences)
token_cnt = Counter()
for i, sentence in enumerate(sentences, 1):
    if i % 500000 == 0:
        print(f'{i/n:.1%}', end=' ', flush=True)
    token_cnt.update(sentence.split())
token_cnt = pd.Series(dict(token_cnt.most_common()))
token_cnt = token_cnt.reset_index()
token_cnt.columns = ['token', 'n']
token_cnt.to_parquet(Path('./Resources/token_cnt.parquet'), compression='UNCOMPRESSED')
# %%
token_by_freq = token_cnt.sort_values(by=['n', 'token'], ascending=[False, True]).token
token2id = {token: i for i, token in enumerate(token_by_freq, 3)}
# %%
from tqdm import tqdm
vector_path = Path('./Resources/Vectors/')
def generate_sequences(min_len=100, max_len=20000, num_words=25000, oov_char=2):
    seq_length = {}
    skipped = 0
    for i, f in tqdm(enumerate((Path('./Resources/ngrams3/')).glob('*.txt'), 1)):
        file_id = f.stem
        text = f.read_text().split('\n')
        vector = [token2id[token] if token2id[token] + 2 < num_words else oov_char 
                  for line in text 
                  for token in line.split()]
        vector = vector[:max_len]
        if len(vector) < min_len:
            skipped += 1
            continue
        seq_length[int(file_id)] = len(vector)
        np.save(Path('./Resources/Vectors/' + str(file_id) + '.npy'), np.array(vector))
    seq_length = pd.Series(seq_length)
    return seq_length
# %%
seq_length = generate_sequences()
pd.Series(seq_length).to_csv(Path('./Resources/seq_length.csv'))
# %%
files = vector_path.glob('*.npy')
filings = sorted([int(f.stem) for f in files])
# %%
prices = pd.read_csv('./Resources/yf_data.csv')
prices.info()
# %%
filing_index = (pd.read_csv(Path('./Resources/10K_10Q_List_With_Tickers.csv'),
                            parse_dates=['Date Filed'])
                .rename(columns=str.lower))
filing_index.info()
# %%
filing_index.head()
# %%
fwd_return = {}
for filing in filings:
    date_filed = filing_index.at[filing, 'date filed']
    price_data = prices[prices.filing==filing].close.sort_index()
    
    try:
        r = (price_data
             .pct_change(periods=5)
             .shift(-5)
             #.loc[:date_filed]
             .iloc[0]
             )
    except:
        continue
    if not np.isnan(r) and -.5 < r < 1:
        fwd_return[filing] = r
len(fwd_return)
# %%
y, X = [], []
for filing_id, fwd_ret in fwd_return.items():
    X.append(np.load(Path('./Resources/Vectors/' + str(filing_id) + '.npy')) + 2)
    y.append(fwd_ret)
y = np.array(y)
len(y), len(X)
# %%
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, GRU, Bidirectional,
                                     Embedding, BatchNormalization, Dropout)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
import tensorflow.keras.backend as K

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
maxlen = 20000
X_train = pad_sequences(X_train, 
                        truncating='pre', 
                        padding='pre', 
                        maxlen=maxlen)

X_test = pad_sequences(X_test, 
                       truncating='pre', 
                       padding='pre', 
                       maxlen=maxlen)
# %%
K.clear_session()
embedding_size = 100
input_dim = X_train.max() + 1
rnn = Sequential([
    Embedding(input_dim=input_dim, 
              output_dim=embedding_size, 
              input_length=maxlen,
             name='EMB'),
    BatchNormalization(name='BN1'),
    Bidirectional(GRU(32), name='BD1'),
    BatchNormalization(name='BN2'),
    Dropout(.1, name='DO1'),
    Dense(5, name='D'),
    Dense(1, activation='linear', name='OUT')
])
# %%
rnn.compile(loss='mse', 
            optimizer='Adam',
            metrics=[RootMeanSquaredError(name='RMSE'),
                     MeanAbsoluteError(name='MAE')])
early_stopping = EarlyStopping(monitor='val_MAE', 
                               patience=5,
                               restore_best_weights=True)
training = rnn.fit(X_train,
                   y_train,
                   batch_size=32,
                   epochs=100,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping],
                   verbose=1)

# %%
vector_path_now = Path('./Resources/Vectors/52084.npy')
vectors = np.load(vector_path_now)
# %%
print(os.getcwd())
# %%
print(vectors)
# %%
