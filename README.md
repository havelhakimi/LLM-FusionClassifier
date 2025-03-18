# LLM-FusionClassifier: for Fine-grained Emotion Classification
We perform Fine-grained Emotion Classification (FEC) on benchmark datasets using `Empathetic Dialogues` and `GoEmotions`. </br>
The feature fusion method is adapted from the **ACL 2024 [paper](https://aclanthology.org/2024.acl-long.433/)** with the corresponding **[GitHub repository](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024)**. 
Instead of fine-tuning LLMs, we extract their representations and train a compact classifier that integrates semantic knowledge and feature interactions through co-occurrence pooling and power normalization. This method ensures an efficient, scalable, and expressive emotion classification pipeline.
