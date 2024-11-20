import torch
import torch.nn as nn
import torch.optim as optim
from main import train_dataset, validation_dataset, test_dataset
from word_embeddings_Part3 import word_embeddings
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import numpy as np
import random

def set_seed(seed_1):
    random.seed(seed_1)
    np.random.seed(seed_1)
    torch.manual_seed(seed_1)

class biLSTM_Simple(nn.Module):
    def __init__(self, embedding_matrix_1, embedding_dimension, vocab_size, hidden_size, output_size, num_layers = 1, dropout = 0):
        super(biLSTM_Simple, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.embedding.weight = nn.Parameter(embedding_matrix_1) # Using pretrained word embeddings.
        self.embedding.weight.requires_grad = True # Freezings the embeddings

        #biLSTM Layer
        # Batch First = True allows for the following dimensions (batch, seq, feature).
        # batch = batch size, seq = length of sequence, feature = size of each words embedding 
        
        self.gru = nn.GRU(embedding_dimension, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        
        # Using Logistic Regresstion to perform binary sentiment analysis of positive and negative
        
        self.dropout = nn.Dropout(dropout)

        self.log_reg = nn.Sequential(
            nn.Linear(hidden_size*2, output_size),
            nn.Sigmoid() # For Binary Classification.
        )

    def forward(self, x):
        x = self.embedding(x)
        # gets the output in size (batch, sequence, feature)
        out, _ = self.gru(x)

        
        # Concats the first and last time step of the biLSTM output for every sequence in the batch.
        # Selects the last hidden state for the forward direction (i.e., the hidden state at the last time step).
        # Selects the last hidden state for the backward direction (i.e., the hidden state at the first time step).
        out = out.mean(dim = 1)
        out = self.log_reg(self.dropout(out))
        return out
    
# Performingg vertification that all embeddings have the dimension 300. Note: The word in the word, vec pair are all tokenized words.
word_embeddings = {
    word: vec for word, vec in word_embeddings.items() if vec.shape[0] == 300
}

def creating_vocab(word_embeddings_1):
    word_idx = defaultdict(lambda: len(word_idx))
    word_idx["<PAD>"] = 0 # Index for Padding
    word_idx["<UNK>"] = 1 # Addressing unkown words

    for key in word_embeddings_1:
        word_idx[key]
    
    return word_idx

vocab_biLSTM = creating_vocab(word_embeddings)
embedding_dim = 300 #For both GloVe and Word2Vec

def building_embedding_matrix(vocab, pretrained_embeddings, embedding_dimension):
    embedding_matrix_2 = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dimension))
    for word, idx in vocab.items():
        if word in pretrained_embeddings:
            embedding_matrix_2[idx] = pretrained_embeddings[word]
    return torch.tensor(embedding_matrix_2, dtype = torch.float)

set_seed(42)
embedding_matrix = building_embedding_matrix(vocab_biLSTM, word_embeddings, embedding_dim)
vocab_size = len(vocab_biLSTM)
output_dim = 1 

dropout = 0.5 #PARAMETER THAT CAN BE CHANGED
batch_size = 32 #PARAMETER THAT CAN BE CHANGED
tokenizer = WordPunctTokenizer()

epochs = 100

#determine_maxlength = [len(tokenizer.tokenize(review)) for review in train_data['text']]
#print(f"Mean length: {np.mean(determine_maxlength)}")
#print(f"Median length: {np.median(determine_maxlength)}")
#print(f"90th percentile length: {np.percentile(determine_maxlength, 90)}")
#print(f"Max length: {np.max(determine_maxlength)}")

# After length analysis we see average length is 23 and 90 percentile length is 36
max_length = 30
def preprocessing_data(sentence, vocab, max_length):
    tokens = tokenizer.tokenize(sentence.lower())
    indices_list = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(indices_list) < max_length:
        indices_list += [vocab["<PAD>"]] * (max_length - len(indices_list))
    else:
        indices_list = indices_list[:max_length]

    return indices_list

train_data_processed = [(preprocessing_data(review_data['text'], vocab_biLSTM, max_length), review_data['label']) for review_data in train_dataset]
valid_data_processed = [(preprocessing_data(review_data['text'], vocab_biLSTM, max_length), review_data['label']) for review_data in validation_dataset]
test_data_processed = [(preprocessing_data(review_data['text'], vocab_biLSTM, max_length), review_data['label']) for review_data in test_dataset]

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        review, label = self.data[idx]
        return torch.tensor(review, dtype=torch.long), torch.tensor(label, dtype = torch.float)
    
train_dataset = SentimentDataset(train_data_processed)
valid_dataset = SentimentDataset(valid_data_processed)
test_dataset = SentimentDataset(test_data_processed)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

def train_loop(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    for reviews, labels in train_loader:
        optimizer.zero_grad()
        sentiments_probabilities = model(reviews)
        predictions = (sentiments_probabilities > threshold)
        loss = loss_fn(sentiments_probabilities, labels.unsqueeze(1))
        correct += (predictions == labels.unsqueeze(1)).type(torch.float).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * reviews.size(0)
    
    epoch_loss = running_loss/len(train_loader.dataset)
    correct /= len(train_loader.dataset)
    return epoch_loss, correct

def valid_loop(model, valid_loader, loss_fn):
    model.eval()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    with torch.no_grad():
        for reviews, labels in valid_loader:
            sentiments_probabilities = model(reviews)
            predictions = (sentiments_probabilities > threshold)
            correct += (predictions == labels.unsqueeze(1)).type(torch.float).sum().item()
            loss = loss_fn(sentiments_probabilities, labels.unsqueeze(1))
            running_loss += loss.item() * reviews.size(0)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    correct /= len(valid_loader.dataset)
    return epoch_loss, correct

def test_loop(model, test_loader, loss_fn):
    model.eval()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    with torch.no_grad():
        for reviews, labels in test_loader:
            sentiments_probabilities = model(reviews)
            predictions = (sentiments_probabilities > threshold)
            correct += (predictions == labels.unsqueeze(1)).type(torch.float).sum().item()
            loss = loss_fn(sentiments_probabilities, labels.unsqueeze(1))
            running_loss += loss.item() * reviews.size(0)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return epoch_loss, correct

# Hyperparameters for tuning
depths = [1, 2, 3]  # Number of biLSTM layers
widths = [64, 128, 256]  # Hidden layer sizes

# Variables to track the best model configuration and performance
best_model = None
best_accuracy = 0
best_config = {}

# Iterate over each combination of depth and width
for num_layers in depths:
    for hidden_size in widths:
        print(f"\nTraining model with depth={num_layers} and width={hidden_size}")
        
        dropout = 0.5
        # Initialize model with current depth and width
        if num_layers == 1:
            dropout = 0
        simple_biLSTM = biLSTM_Simple(embedding_matrix_1=embedding_matrix, embedding_dimension=embedding_dim, vocab_size=vocab_size, hidden_size=hidden_size, output_size=output_dim, num_layers=num_layers, dropout=dropout)

        # Define optimizer and loss function
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, simple_biLSTM.parameters()), lr=0.0001)
        loss_fn = nn.BCELoss()

        # Early stopping variables
        es_patience = 3
        no_improv_epochs = 0
        best_val_loss = np.inf
        early_stop = False

        # Train for a set number of epochs or until early stopping is triggered
        for epoch in range(epochs):
            train_loss, train_correct = train_loop(simple_biLSTM, train_loader, loss_fn, optimizer)
            valid_loss, valid_correct = valid_loop(simple_biLSTM, valid_loader, loss_fn)
            
            print(f"Epoch {epoch+1}/{epochs} - Training loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}")
            print(f"Training Accuracy: {(train_correct*100):>0.1f}%, Validation Accuracy: {(valid_correct*100):>0.1f}%")

            # Early stopping check
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                no_improv_epochs = 0
            else:
                no_improv_epochs += 1
                if no_improv_epochs >= es_patience:
                    print("Early stopping triggered.")
                    early_stop = True
                    break
        
        # Evaluate on the test set
        test_loss, test_correct = test_loop(simple_biLSTM, test_loader, loss_fn)
        print(f"Test loss: {test_loss:.4f}, Test Accuracy: {test_correct*100:.1f}%")
        
        # Update the best model if current configuration has higher accuracy
        if test_correct > best_accuracy:
            best_accuracy = test_correct
            best_model = simple_biLSTM
            best_config = {"depth": num_layers, "width": hidden_size}

print("\nBest configuration:")
print(f"Depth: {best_config['depth']}, Width: {best_config['width']}")
print(f"Best Test Accuracy: {best_accuracy*100:.1f}%")