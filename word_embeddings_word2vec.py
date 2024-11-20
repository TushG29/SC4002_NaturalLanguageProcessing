import torch
from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.tokenize import WordPunctTokenizer
from main import train_dataset
import pandas as pd

# Create and train Word2Vec model 
word2vec_model = api.load('word2vec-google-news-300')

# Create dictionary for word embeddings
word_embeddings = {}            # Dictionary for word embeddings

# Store unique words
vocab = set()  
oov_words_Word2Vec = set()      # Set to store OOV words for Word2Vec

tokenizer = WordPunctTokenizer()    

# Function to check if word is in Word2Vec model
def Word2Vec_Embedder(word):
    if word in word2vec_model:
        word_embeddings[word] = word2vec_model[word]
        return True
    
    else:
        oov_words_Word2Vec.add(word)
        return False


for example in train_dataset:
    
    text = example['text']
    words = tokenizer.tokenize(text)

    for word in words:

        word = word.lower()
        
        # Add words to vocab
        vocab.add(word)

        # Check if word is in Word2Vec model
        if Word2Vec_Embedder(word):
            pass
        
        else:
            word_embeddings[word] = torch.rand(300)
        
                 
# Save the word embeddings
torch.save(word_embeddings, 'word_embedding_word2vec.pth')


vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')
print(f'OOV words Word2vec: {len(oov_words_Word2Vec)}')

'''
list = list(oov_words_Word2Vec)

df = pd.DataFrame(list, columns=['OOV_words']) 

df.to_excel("Word2VecOOV.xlsx", index=False)
'''