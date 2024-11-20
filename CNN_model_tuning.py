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

embeddings_path = 'word_embedding_updated.pth'

embedding_dim = 300

# Load pre-trained word embeddings
word_embeddings = torch.load(embeddings_path)


def set_seed(seed_1):
    random.seed(seed_1)
    np.random.seed(seed_1)
    torch.manual_seed(seed_1)



class CNN_Simple(nn.Module):
    def __init__(self, embedding_matrix, embedding_dimension, vocab_size, output_size, hidden_size, num_layers, dropout_rate):
        super(CNN_Simple, self).__init__()  # Corrected super call

        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = True

        # Create multiple layers of convolutions
        self.conv_layers = nn.ModuleList()
        for depth in range(num_layers):
            if depth == 0:
                # First layer takes input from embedding
                layer_convs = nn.ModuleList([
                    nn.Conv2d(1, hidden_size, (k, embedding_dimension))
                    for k in kernel_sizes
                ])
            else:
                # Subsequent layers take input from previous layer
                layer_convs = nn.ModuleList([
                    nn.Conv2d(hidden_size * len(kernel_sizes), hidden_size, (k, 1))
                    for k in kernel_sizes
                ])
            self.conv_layers.append(layer_convs)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * len(kernel_sizes), output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)     # [batch_size, 1, seq_len, embedding_dim]

        # Process through each convolutional layer
        for depth, conv_layer in enumerate(self.conv_layers):
            if depth > 0:
                x = x.unsqueeze(1)  # Add dummy dimension for convolution

            # Apply each kernel size convolution
            conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in conv_layer]
            pooled_outputs = [torch.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]
            x = torch.cat(pooled_outputs, 1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def build_embedding_matrix(vocab, pretrained_embeddings, embedding_dimension):
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dimension))
    for word, idx in vocab.items():
        if word in pretrained_embeddings:
            embedding_matrix[idx] = pretrained_embeddings[word]
    return torch.tensor(embedding_matrix, dtype=torch.float)


# Create vocabulary and embedding matrix
def creating_vocab(word_embeddings):
    word_idx = defaultdict(lambda: len(word_idx))
    word_idx["<PAD>"] = 0
    word_idx["<UNK>"] = 1

    for word in word_embeddings:
        word_idx[word]
    return word_idx

vocab_cnn = creating_vocab(word_embeddings)

max_length = 30
set_seed(42)
dropout_rate = 0.5
embedding_matrix = build_embedding_matrix(vocab_cnn, word_embeddings, embedding_dim)
vocab_size = len(vocab_cnn)
output_dim = 1
num_filters = 100
kernel_sizes = [3, 4, 5]
batch_size = 128

tokenizer = WordPunctTokenizer()
epochs = 100

depths = [1]
widths = [64, 128, 256]


# Tracking best model configuration
best_model = None
best_accuracy = 0
best_config = {}

def preprocessing_data(sentence, vocab, max_length):
    tokens = tokenizer.tokenize(sentence.lower())
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(indices) < max_length:
        indices += [vocab["<PAD>"]] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    return indices

train_data_processed = [(preprocessing_data(review['text'], vocab_cnn, max_length), review['label']) for review in train_dataset]
valid_data_processed = [(preprocessing_data(review['text'], vocab_cnn, max_length), review['label']) for review in validation_dataset]
test_data_processed = [(preprocessing_data(review['text'], vocab_cnn, max_length), review['label']) for review in test_dataset]

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review, label = self.data[idx]
        return torch.tensor(review, dtype=torch.long), torch.tensor(label, dtype=torch.float)

train_loader = DataLoader(SentimentDataset(train_data_processed), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(SentimentDataset(valid_data_processed), batch_size=batch_size)
test_loader = DataLoader(SentimentDataset(test_data_processed), batch_size=batch_size)

# Training, Validation, and Testing Loops
def train_loop(model, train_loader, loss_fn, optimizer, clip_value=1.0):
    model.train()
    running_loss, correct = 0.0, 0
    for reviews, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(reviews)
        loss = loss_fn(outputs, labels.unsqueeze(1))
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        running_loss += loss.item() * reviews.size(0)
        correct += ((outputs > 0.5) == labels.unsqueeze(1)).type(torch.float).sum().item()

    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

def valid_loop(model, valid_loader, loss_fn):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for reviews, labels in valid_loader:
            outputs = model(reviews)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            running_loss += loss.item() * reviews.size(0)
            correct += ((outputs > 0.5) == labels.unsqueeze(1)).type(torch.float).sum().item()
    return running_loss / len(valid_loader.dataset), correct / len(valid_loader.dataset)

def test_loop(model, test_loader, loss_fn):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            outputs = model(reviews)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            running_loss += loss.item() * reviews.size(0)
            correct += ((outputs > 0.5) == labels.unsqueeze(1)).type(torch.float).sum().item()
    return running_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# Iterate over each combination of depth and width
for num_layers in depths:
    for hidden_size in widths:
        print(f"\nTraining model with depth={num_layers} and width={hidden_size}")

        # Initialize model with current depth and width
        model = CNN_Simple(
            embedding_matrix=embedding_matrix,  # Corrected to match the constructor argument name
            embedding_dimension=embedding_dim,
            vocab_size=vocab_size,
            output_size=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate  # Include dropout_rate if needed
          )

        loss_fn = nn.BCEWithLogitsLoss() #Check Sigmoid layer and the BCELoss in one single class
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)



        # Early stopping variables
        es_patience = 5
        no_improv_epochs = 0
        best_val_loss = np.inf
        early_stop = False

        # Train for a set number of epochs or until early stopping is triggered
        for epoch in range(epochs):
            train_loss, train_correct = train_loop(model, train_loader, loss_fn, optimizer)
            valid_loss, valid_correct = valid_loop(model, valid_loader, loss_fn)

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
        test_loss, test_correct = test_loop(model, test_loader, loss_fn)
        print(f"Test loss: {test_loss:.4f}, Test Accuracy: {test_correct*100:.1f}%")

        # Update the best model if current configuration has higher accuracy
        if test_correct > best_accuracy:
            best_accuracy = test_correct
            best_model = model
            best_config = {"depth": num_layers, "width": hidden_size}

print("\nBest configuration:")
print(f"Depth: {best_config['depth']}, Width: {best_config['width']}")
print(f"Best Test Accuracy: {best_accuracy*100:.2f}%")
