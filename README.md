# LLM-FusionClassifier: for Fine-grained Emotion Classification
- We perform Fine-grained Emotion Classification (FEC) on benchmark datasets using `Empathetic Dialogues(ED)` and `GoEmotions`.
- Our approach leverages feature extraction from three large language models (LLMs)—**Llama-2-7b-chat**, **BERT-large**, and **RoBERTa-large**—followed by a structured feature fusion and a lightweight classification model.  The feature fusion method is adapted from the ACL 2024 paper [LLMEmbed](https://aclanthology.org/2024.acl-long.433/) with the corresponding [GitHub repository](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024).
- Instead of fine-tuning LLMs, we extract their representations and train a compact classifier that integrates semantic knowledge and feature interactions through co-occurrence pooling and power normalization. This method ensures an efficient, scalable, and expressive emotion classification pipeline.

## Feature Extraction for LLaMA2, BERT-large, and RoBERTa-large
### For Llama2
- The code for creating and saving feature tensors is in `llama2_rep_extract.py`. The data for ED and go_emotion is `data` folder.
- To extract and save LLaMA2 feature tensors for the training set on ED dataset, run the following:
  - ```python llama2_rep_extract.py -device cuda -task ED -mode train```
- Some imp arguments : `-task` denotes name of dataset possible options are `ED` and `go_emotion`. `-mode` Possible options are `train`, `test` and `valid`.
- For LLaMA2, the possible model choices are the base model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and the version optimized for dialogue use cases, [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). We used `Llama-2-7b-chat-hf` because `Llama-2-7b-hf` produced NaN values on the validation and test sets.
- For LLaMA2, the embedding dimension is 4096. We average the embeddings across all tokens in the last 5 layers and save the resulting tensor of shape (5, 4096) per sample, where 5 represents the last 5 layers. For reference, see the log file `Create_data_LLAMA2_GO.out` in the `Logs` folder.

### For BERT-large and Roberta-large
- The code for creating and saving feature tensors is in `X_rep_extract.py` where X is bert/roberta. The data for ED and go_emotion is `data` folder.
- To extract and save bert feature tensors for the training set on ED dataset, run the following:
  - ```python bert_rep_extract.py -device cuda -task ED -mode train```
- For BERT/RoBERTa, the embedding dimension is 1024. We save the representation of the [CLS] token (i.e., the 0th token) from the final layer. For each sample, we store a tensor of shape (1024). For reference, see the log files in the `Logs` folder.
