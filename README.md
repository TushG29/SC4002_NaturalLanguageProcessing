# SC4002_NaturalLanguageProcessing
SC4002 Natural Language Processing Project


Run the following files in order

Q1:
  Run main.py file first
  Run both word_embeddings_GloVe.py and word_embeddings_word2vec.py files

Explanations of sample outputs:
  1. a) shows the output size of vocab formed using our training data based on the chosen embedding (Word2Vec is 16545), (Glove is 18950).
  1. b) shows the number of OOV words that exist in our training data with the respective embedding (Word2Vec is 1651), (Glove is 3034). 


Q2: 
  Run rnn_model_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = True is put to False in line 23.
  Run max_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = True is put to False in line 23.
  Run mean_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = True is put to False in line 23.
  Run model_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = True is put to False in line 36.

Q2 Sample output (full results in the report):
| Depth | Width | Model Performance                                                       |
|-------|-------|-------------------------------------------------------------------------|
| 1     | 256   | Test loss: 0.5293, Test Accuracy: 74.6%, Early Stopping after 20 epochs |


Q3:
  Run rnn_model_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = False is put to True in line 23.
  Run max_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = False is put to True in line 23.
  Run mean_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = False is put to True in line 23.
  Run model_pooling_tuning_word2vec.py. Ensure that self.embedding.weight.requires_grad = False is put to True in line 36.
  
  Run rnn_model_tuning_updated_embeddings.py. Ensure that self.embedding.weight.requires_grad = True is put to True in line 23.
  Run max_pooling_tuning_updated_embeddings.py. Ensure that self.embedding.weight.requires_grad = True is put to True in line 23.
  Run mean_pooling_tuning_updated_embeddings.py. Ensure that self.embedding.weight.requires_grad = True is put to True in line 23.
  Run model_pooling_tuning_updated_embeddings.py. Ensure that self.embedding.weight.requires_grad = True is put to True in line 36.

  Run biLSTM_model.py
  Run biLSTM_max.py
  Run biLSTM_mean.py
  Run biLSTM_attention.py

  Run biGRU_model.py
  Run biGRU_max.py
  Run biGRU_mean.py
  Run biGRU_attention.py
  *Note: manually change batch size in the biLSTM and biGRU files and re-run*

  Run the CNN_model_tuning.py

  Sample Output:
  | Experiment                                | Model Performance                                                | Model Details                                                                                               |
|-------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| rnn_model_tuning_updated_embeddings        | Test Accuracy: 79.1%<br>Test Loss: 0.5894<br>Early stopping after 12 epochs | Depth: 1<br>Width: 64<br>Max Epochs: 100<br>Learning Rate: Default<br>Optimizer: Adam<br>Batch Size: 64     |
| Model_attention_tuning_updated_embeddings  | Test Accuracy: 77.5%<br>Test Loss: 0.6115<br>Early Stopping after 10 epochs | Depth: 1<br>Width: 64<br>Max Epochs: 100<br>Learning Rate: Default<br>Optimizer: Adam<br>Batch Size: 64     |
| Mean_pooling_tuning_updated_embeddings     | Test Accuracy: 76.5%<br>Test Loss: 0.6099                         | Depth: 1<br>Width: 64<br>Max Epochs: 100                                                                    |


  
