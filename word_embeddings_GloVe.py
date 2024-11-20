import torch
from torchtext.vocab import GloVe
from nltk.tokenize import WordPunctTokenizer
from main import train_dataset
import pandas as pd

# Load GloVe embeddings
glove = GloVe(name='6B', dim=300)

# Create dictionary for word embeddings
word_embeddings = {}            # Dictionary for word embeddings

# Store unique words
vocab = set()  
oov_words_GloVe = set()      # Set to store OOV words for Word2Vec

tokenizer = WordPunctTokenizer()    

# Function to check if word is in GloVe model
def GloVe_embedder(word):
    if word in glove.stoi:
        word_embeddings[word] = glove[word]
        return True
        
    else:
        oov_words_GloVe.add(word)
        return False
    

for example in train_dataset:
    text = example['text']
    words = text.split()

    for word in words:
        word = word.lower()  # Convert to lowercase to match GloVe vocab
        
        # Add word to vocab
        vocab.add(word)

        # Check if word is in GloVe
        if GloVe_embedder(word):
            pass
        
        else:
            word_embeddings[word] = torch.rand(300)


# Save the word embeddings
torch.save(word_embeddings, 'word_embeddings_GloVe.pth')

# Output the results
vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')
print(f'OOV words in GloVe: {len(oov_words_GloVe)}')

list = list(oov_words_GloVe)

df = pd.DataFrame(list, columns=['OOV_words']) 

df.to_excel("GloVeOOV.xlsx", index=False)


