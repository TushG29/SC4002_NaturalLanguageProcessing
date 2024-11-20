# Welcome to our SC4002 Natural Language Processing Course Project

## About
This is a project for SC4002 (Natural Language Processing), where we worked on performing sentiment classification.  
Follow the instructions below to run the project files.

---

## Q1: Embedding Preparation

### Steps to Run:
1. Run `main.py`.
2. Run `word_embeddings_GloVe.py`.
3. Run `word_embeddings_word2vec.py`.

### Sample Outputs:
1. **Vocabulary Size**:
   - Word2Vec: 16,545  
   - GloVe: 18,950  
2. **Out-of-Vocabulary (OOV) Words**:
   - Word2Vec: 1,651  
   - GloVe: 3,034  

---

## Q2: RNN and Pooling with Word2Vec Embeddings (Static)

### Steps to Run:
1. Run `rnn_model_tuning_word2vec.py`  
   Ensure `self.embedding.weight.requires_grad = False` (line 23).  
2. Run `max_pooling_tuning_word2vec.py`  
   Ensure `self.embedding.weight.requires_grad = False` (line 23).  
3. Run `mean_pooling_tuning_word2vec.py`  
   Ensure `self.embedding.weight.requires_grad = False` (line 23).  
4. Run `model_pooling_tuning_word2vec.py`  
   Ensure `self.embedding.weight.requires_grad = False` (line 36).  

### Sample Output:
| Depth | Width | Model Performance                                                       |
|-------|-------|-------------------------------------------------------------------------|
| 1     | 256   | Test loss: 0.5293, Test Accuracy: 74.6%, Early Stopping after 20 epochs |

---

## Q3: Advanced Models and Updated Embeddings

### Steps to Run:
1. **RNN and Pooling with Word2Vec Embeddings (Trainable)**:  
   - Run `rnn_model_tuning_word2vec.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  
   - Run `max_pooling_tuning_word2vec.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  
   - Run `mean_pooling_tuning_word2vec.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  
   - Run `model_pooling_tuning_word2vec.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 36).  

2. **Updated Embeddings with Attention and Pooling**:  
   - Run `rnn_model_tuning_updated_embeddings.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  

     **Output**:  
     - Vocabulary size: 16,545  
     - OOV Words:  
       - Word2Vec: 1,651  
       - FastText: 386  
       - Lemmatizer: 384  
       - Digits Embedder: 263  
       - Edit Distance: 66  

   - Run `max_pooling_tuning_updated_embeddings.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  
   - Run `mean_pooling_tuning_updated_embeddings.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 23).  
   - Run `model_pooling_tuning_updated_embeddings.py`  
     Ensure `self.embedding.weight.requires_grad = True` (line 36).  

3. **LSTM and GRU Models**:  
   - Run `biLSTM_model.py`  
   - Run `biLSTM_max.py`  
   - Run `biLSTM_mean.py`  
   - Run `biLSTM_attention.py`  
   - Run `biGRU_model.py`  
   - Run `biGRU_max.py`  
   - Run `biGRU_mean.py`  
   - Run `biGRU_attention.py`  

   **Note**: Manually change the batch size in the biLSTM and biGRU files and re-run.  

4. **CNN Model**:  
   - Run `CNN_model_tuning.py`.  

### Sample Output:
| Experiment                                | Model Performance                                                | Model Details                                                                                               |
|-------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| rnn_model_tuning_updated_embeddings        | Test Accuracy: 79.1%<br>Test Loss: 0.5894<br>Early stopping after 12 epochs | Depth: 1<br>Width: 64<br>Max Epochs: 100<br>Learning Rate: Default<br>Optimizer: Adam<br>Batch Size: 64     |
| Model_attention_tuning_updated_embeddings  | Test Accuracy: 77.5%<br>Test Loss: 0.6115<br>Early Stopping after 10 epochs | Depth: 1<br>Width: 64<br>Max Epochs: 100<br>Learning Rate: Default<br>Optimizer: Adam<br>Batch Size: 64     |
| Mean_pooling_tuning_updated_embeddings     | Test Accuracy: 76.5%<br>Test Loss: 0.6099                         | Depth: 1<br>Width: 64<br>Max Epochs: 100                                                                    |

---

Feel free to refer to the report for full experimental results and analysis.
