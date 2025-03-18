# LLM-FusionClassifier: for Fine-grained Emotion Classification
- We perform Fine-grained Emotion Classification (FEC) on benchmark datasets using `Empathetic Dialogues(ED)` and `GoEmotions`.
- Our approach leverages feature extraction from three large language models (LLMs)—**Llama-2-7b-chat**, **BERT-large**, and **RoBERTa-large**—followed by a structured feature fusion and a lightweight classification model.  The feature fusion method is adapted from the ACL 2024 paper [LLMEmbed](https://aclanthology.org/2024.acl-long.433/) with the corresponding [GitHub repository](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024).
- Instead of fine-tuning LLMs, we extract their representations and train a compact classifier that integrates semantic knowledge and feature interactions through co-occurrence pooling and power normalization. This method ensures an efficient, scalable, and expressive emotion classification pipeline.

## Feature Extraction for LLaMA2, BERT-large, and RoBERTa-large
- The code for creating and saving feature tensors is in `X_rep_extract.py`, where `X` represents `llama2`, `bert`, or `roberta`. 
- The data for ED and go_emotion is `data` folder. For example, to extract and save LLaMA2 feature tensors for the training set on ED dataset, run the following:
  - ```python llama2_rep_extract.py -device cuda -task ED -mode train```
- Some imp arguments : `-task` denotes name of dataset possible options are `ED` and `go_emotion`. `-mode` Possible options are `train`, `test` and `valid`.
-For LLaMA2, the possible model choices are the base model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and the version optimized for dialogue use cases, [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). We used `Llama-2-7b-chat-hf` because `Llama-2-7b-hf` produced NaN values on the validation and test sets.

