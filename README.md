# LLM-FusionClassifier: for Fine-grained Emotion Classification
We perform Fine-grained Emotion Classification (FEC) on benchmark datasets using `Empathetic Dialogues` and `GoEmotions`.
Our approach leverages feature extraction from three large language models (LLMs)—LLaMA2, BERT-large, and RoBERTa-large—followed by a structured feature fusion and a lightweight classification model. The feature fusion method is adapted from the paper Insert Paper Title.
Instead of fine-tuning LLMs, we extract their representations and train a compact classifier that integrates semantic knowledge and feature interactions through co-occurrence pooling and power normalization. This method ensures an efficient, scalable, and expressive emotion classification pipeline.
