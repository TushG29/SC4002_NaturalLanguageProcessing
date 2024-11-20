import torch
import gensim.downloader as api
from torchtext.vocab import FastText
from main import train_dataset
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import Levenshtein
import pandas as pd

nltk.download('wordnet')

# Define the edit distance threshold
EDIT_DISTANCE_THRESHOLD = 2


# Create and train Word2Vec model 
word2vec_model = api.load('word2vec-google-news-300')
fasttext = FastText(language='en')

# Create dictionary for word embeddings
word_embeddings = {}            # Dictionary for word embeddings
seen_words_EditDistance = {}    # Dictionary to store words that have had their edit distance calculated and assigned a word in the word2vec/fasttext model

# Store unique words
vocab = set()  
oov_words_Word2Vec = set()      # Set to store OOV words for Word2Vec
oov_words_FastText = set()      # Set to store OOV for FastText
oov_words_lemma = set()         # Set to store OOV words for Lemmatization
oov_words_digit_embedder = set()
oov_words_EditDistance = set()  # Set to store OOV words for Edit Distance

tokenizer = WordPunctTokenizer()    
lemmatizer = WordNetLemmatizer()

#Number Embedding Dictionary
number_embeddings = {"1": torch.rand(300), "2": torch.rand(300), "3": torch.rand(300), "4": torch.rand(300), "5": torch.rand(300), 
                     "6": torch.rand(300), "7": torch.rand(300), "8": torch.rand(300), "9": torch.rand(300), "0": torch.rand(300)}


#Function to remove words that are only symbols
def remove_symbol_words(word):
    return any(char.isalnum() for char in word)


# Function to check if word is in Word2Vec model
def Word2Vec_Embedder(word):
    if word in word2vec_model:
        word_embeddings[word] = word2vec_model[word]
        return True
    
    else:
        oov_words_Word2Vec.add(word)
        return False


# Function to check if word is in FastText
def FastText_Embedder(word):
    if word in fasttext.stoi:
        word_embeddings[word] = fasttext[word]
        return True
    
    else:
        oov_words_FastText.add(word)
        return False


# Function to calculate Edit Distance and check if in Word2Vec
def find_closest_word_word2vec(word, threshold):
    if word in seen_words_EditDistance:
        closest_word = seen_words_EditDistance[word]
        word_embeddings[word] = closest_word
        return True
    
    closest_word = None
    closest_distance = float('inf')

    # Iterate through words in the model vocabulary
    for vocab_word in word2vec_model.key_to_index:
        distance = Levenshtein.distance(word, vocab_word)
        if distance < closest_distance and distance <= threshold:
            closest_word = vocab_word
            closest_distance = distance
    
    if closest_word:
        seen_words_EditDistance[word] = word2vec_model[closest_word]
        word_embeddings[word] = word2vec_model[closest_word]
        return True
    
    else:
        oov_words_EditDistance.add(word)
        return False


# Function to calculate Edit Distance and check if in FastText
def find_closest_word_FastText(word, threshold):
    closest_word = None
    closest_distance = float('inf')
    
    if word in seen_words_EditDistance:
        closest_word = seen_words_EditDistance[word]
        word_embeddings[word] = closest_word
        return True

    # Iterate through words in the model vocabulary
    for vocab_word in fasttext.stoi:
        distance = Levenshtein.distance(word, vocab_word)
        if distance < closest_distance and distance <= threshold:
            closest_word = vocab_word
            closest_distance = distance
    
    if closest_word:
        seen_words_EditDistance[word] = fasttext[closest_word]
        word_embeddings[word] = fasttext[closest_word]
        return True
    
    else:
        oov_words_EditDistance.add(word)
        return False


def embed_digits(number):
    check_cycle = len(number)
    counter=0
    digit_list=[]
    for digit in number:
        if digit in number_embeddings:
            digit_list.append(digit)
            counter+=1
            pass
        else:
            oov_words_digit_embedder.add(number)
            return False
    
    if counter==check_cycle:
        for embeddings in digit_list:
            word_embeddings[embeddings] = number_embeddings[embeddings]
            return True
    


#Function for Lemmatizer
# Lemmatize and see if the word is not in FastText
def Lemmatized_Embedder(word):
    lemma = lemmatizer.lemmatize(word)
    if lemma in word2vec_model:
        word_embeddings[word] = word2vec_model[lemma]
        return True
        
    elif lemma in fasttext.stoi:
        word_embeddings[word] = fasttext[lemma]
        return True
        
    else:
        oov_words_lemma.add(word)
        return False
    


for example in train_dataset:
    
    text = example['text']
    words = tokenizer.tokenize(text)

    for word in words:

        word = word.lower()
        
        # Add words to vocab
        vocab.add(word)

        # Check if word is in Word2Vec model
        # WONT THIS ADD DUPLICATED WORD EMBEDDINGS
        if Word2Vec_Embedder(word):
            pass
        
        elif FastText_Embedder(word):
            pass
        
        elif Lemmatized_Embedder(word):
            pass
        
        elif embed_digits(word):
            pass
        
        elif not remove_symbol_words(word):
            pass
                
        elif find_closest_word_word2vec(word, EDIT_DISTANCE_THRESHOLD):
            pass
        
        elif find_closest_word_FastText(word, EDIT_DISTANCE_THRESHOLD):
            pass
        
        else:
            word_embeddings[word] = torch.rand(300)
        
                 
# Save the word embeddings
torch.save(word_embeddings, 'word_embedding_updated.pth')


vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')
print(f'OOV words Word2vec: {len(oov_words_Word2Vec)}')
print(f'OOV words FastText: {len(oov_words_FastText)}')
print(f'OOV words Lemmatizer: {len(oov_words_lemma)}')
print(f'OOV words Digits Embedder: {len(oov_words_digit_embedder)}')
print(f'OOV words Edit Distance: {len(oov_words_EditDistance)}')

'''
list_1 = list(oov_words_Word2Vec)
list_2 = list(oov_words_FastText)
list_3 = list(oov_words_lemma)
list_4 = list(oov_words_digit_embedder)
list_5 = list(oov_words_EditDistance)

df1 = pd.DataFrame(list_1, columns=['OOV_words']) 
df2 = pd.DataFrame(list_2, columns=['OOV_words'])
df3 = pd.DataFrame(list_3, columns= ['OOV_Words'])
df4 = pd.DataFrame(list_4, columns=["OOV_Words"])
df5 = pd.DataFrame(list_5, columns=["OOV_Words"])

df1.to_excel("Word2VecOOV.xlsx", index=False)
df2.to_excel("FastTextOOV.xlsx", index=False)
df3.to_excel("LemmaOOV.xlsx", index=False)
df4.to_excel("DigitsOOV.xlsx", index=False)
df5.to_excel("FinalOOV.xlsx", index=False)

print("done")
'''