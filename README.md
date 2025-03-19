# LLM-FusionClassifier: for Fine-grained Emotion Classification (Single-label classification)
- We perform Fine-grained Emotion Classification (FEC) on benchmark datasets using `Empathetic Dialogues(ED)` (32 labels) and `GoEmotions` (27 labels).
- Our approach leverages feature extraction from three large language models (LLMs)—**Llama-2-7b-chat**, **BERT-large**, and **RoBERTa-large**—followed by a structured feature fusion and a lightweight classification model.  The feature fusion method and the classifier is adapted from the ACL 2024 paper [LLMEmbed](https://aclanthology.org/2024.acl-long.433/) with the corresponding [GitHub repository](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024).
- Instead of fine-tuning LLMs, we extract their representations and train a compact classifier that integrates semantic knowledge and feature interactions through co-occurrence pooling and power normalization. This method ensures an efficient, scalable, and expressive emotion classification pipeline.

## Feature Extraction for LLaMA2, BERT-large, and RoBERTa-large
### For Llama2
- The code for creating and saving feature tensors is in `llama2_rep_extract.py`. The data for ED and go_emotion is `data` folder.
- To extract and save LLaMA2 feature tensors for the training set on ED dataset, run the following:
  - ```python llama2_rep_extract.py -device cuda -task ED -mode train```
- Some imp arguments : `-task` denotes name of dataset possible options are `ED` and `go_emotion`. `-mode` Possible options are `train`, `test` and `valid`.
- For ED, the tensors will be saved inside 'dataset\ED\llama2_7b_chat\`. Similarly for go_emotion. A subset of the first three samples is in the `data\ED\dataset_tensor` folder for the train, test, and validation datasets. 
- For LLaMA2, the possible model choices are the base model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and the version optimized for dialogue use cases, [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). We used `Llama-2-7b-chat-hf` because `Llama-2-7b-hf` produced NaN values on the validation and test sets.
- For LLaMA2, the embedding dimension is 4096. We average the embeddings across all tokens in the last 5 layers and save the resulting tensor of shape (5, 4096) per sample, where 5 represents the last 5 layers. For reference, see the log file `Create_data_LLAMA2_GO.out` in the `Logs` folder.

### For BERT-large and Roberta-large
- The code for creating and saving feature tensors is in `X_rep_extract.py` where X is bert/roberta. The data for ED and go_emotion is `data` folder.
- To extract and save bert feature tensors for the training set on ED dataset, run the following:
  - ```python bert_rep_extract.py -device cuda -task ED -mode train```
- For ED, the tensors will be saved inside 'dataset\ED\bert\`. Likewise the tensors in case of roberta will be saved inside 'dataset\ED\roberta\`. Similarly it will be saved for go_emotion. A subset of the first three samples is in the `data\ED\dataset_tensor` folder for the train, test, and validation datasets.
- For BERT/RoBERTa, the embedding dimension is 1024. We save the representation of the [CLS] token (i.e., the 0th token) from the final layer. For each sample, we store a tensor of shape (1024). For reference, see the log files in the `Logs` folder.

## Run the LLM-fusion classifier
After feature extraction and saving tensors inside the `data` folder, run the following script to train the FEC on ED dataset. </br>
```python main.py -dataset 'ED' ``` </br>
There are a few other optional runtime arguments, which can be found in `main.py`

## Results on test set
We report the average results over 5 independent random runs.
### For ED: Acc: 60.44±0.47; weighted-F1 score: 59.80±0.54
### For go_emotion: Acc: 58.13±0.40; weighted-F1 score: 57.54±0.23
The results on the ED dataset are comparable to [SOTA](https://aclanthology.org/2023.acl-long.613/) approaches, whereas for GoEmotions, the performance is significantly below the current SOTA. The Empathetic Dialogues (ED) dataset consists of conversational dialogues, which align well with the underlying LLaMA-2-7B-Chat model, as it is also fine-tuned for dialogue-based use cases. The GoEmotions (go_emotions) dataset consists of Reddit comments, which differ in style and structure from conversational dialogues. Since LLaMA-2-7B-Chat is optimized for dialogues, this domain mismatch likely contributes to its lower performance on GoEmotions.
## Explanation of Structured Feature Fusion and Co-occurrence Pooling Used in the LLM-Fusion Classifier
The code for this module is in `DownstreamModel.py`
### 1. Compression of LLaMA2 Features
- LLaMA2 produces (5, 4096) embeddings per sample.
- Each 4096-d embedding is compressed to 1024-d using a linear transformation (`4096 → 1024`).
- This results in a compressed representation of shape (5, 1024).

### 2. Concatenation with BERT and RoBERTa Features
- The compressed LLaMA2 embeddings (5, 1024) are combined with:
  - BERT CLS embedding (1024)
  - RoBERTa CLS embedding (1024)
- The final representation is stacked into a tensor of shape (7, 1024) per sample.

### 3. Co-occurrence Pooling for Feature Interaction
- Instead of simple concatenation, we compute pairwise feature interactions.
- The interaction matrix (7×7) is computed using dot products.
- This is then flattened into a vector of size 49.

### 4. Power Normalization for Scaling
- A nonlinear transformation (`tanh(2*sigma*X)`) is applied to rescale interaction values to (-1,1). sigma is a hyperparameter that can be tuned. We have used a default value of 1.
- This ensures balanced feature magnitudes.

### 5. Fusion with LLaMA2 Mean Representation
- The mean-pooled LLaMA2 embedding (4096-d) is computed.
- The final representation is concatenated as `[interaction_features (49) + mean_LLaMA2 (4096)]`.
- This results in a final feature vector of size (4145-d) per sample.

### 6. Feedforward network for classification 
- A small feedforward network maps features to logit scores, which are used for classification.





