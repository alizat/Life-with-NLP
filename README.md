# Life-with-NLP
Pet projects involving NLP tasks

Done so far:
- [Classifying Emotions](classifying_emotions.ipynb)
  - Text tokenization, embeddings, and classification with DistilBERT + fine-tuning on the [Emotions dataset](https://huggingface.co/datasets/emotion)
- [Encoders & Decoders](encoder_decoder.ipynb)
  - Walkthrough explaining the structure of Transformers and encoder-decoder-style neural networks
- [Document Summarization](summarization.ipynb) (In Progress)
  - Demonstration of summarization via some of the popular transformer models out there as well as a couple of evaluation metrics that are used to model performance on summarization tasks
- [Transformer Pipelines](transformer_pipelines.ipynb)
  - A tour of the *pipelines* offered by the Transformers library for standard NLP tasks such as text generation, summarization, named entity recognition, etc.
- [Multi-class Classification of Arabic Documents](arabic_document_classification/) (In Progress)
  - Predict document type out of 5 different types for > 100K Arabic texts.
  - Multiple ways to predict document types: 
    - Train basic ML classifier on static embeddings
    - Predict with NLP models fine-tuned on the data
