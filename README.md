# GPT Implementation in PyTorch

This repository contains a from-scratch implementation of the GPT (Generative Pre-trained Transformer) architecture using PyTorch. The implementation focuses on understanding the core components of the transformer architecture and its application to language modeling.

## Overview

GPT (Generative Pre-trained Transformer) represents a significant advancement in natural language processing. This implementation breaks down the key components:

- Token and positional embeddings
- Multi-headed self-attention mechanism  
- Feed-forward neural networks
- Transformer blocks
- Output projection layer

The model is trained on a corpus of text (in this case, Alice in Wonderland) to predict the next token in a sequence.

## Architecture Details

The implementation includes several key components:

### Tokenization
- Word-level tokenization
- Vocabulary creation with special tokens (`<PAD>`, `<UNK>`, `<START>`, `<END>`)
- Token-to-index mapping for model input

### Model Components
- **Multi-Head Attention**: Allows the model to focus on different aspects of the input sequence simultaneously
- **Feed-Forward Networks**: Processes each position's representations independently  
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Positional Embeddings**: Provides position information to the model
- **Transformer Blocks**: Combines attention and feed-forward networks with residual connections

### Key Features
- Autoregressive text generation
- Configurable model size (layers, embedding dimension, heads)
- Temperature-controlled sampling
- Optional top-k sampling for better generation quality


## Model Configuration

The default configuration for this implementation:

```python
config = GPTConfig(
   vocab_size=2000, 
   block_size=128,
   n_layer=6,
   n_embd=384,
   num_heads=6,
   dropout=0.1
)
```

## Notebook Explanation
We've also included a notebook `GPT_Implementation.ipynb` which explains all aspects of the model architecture for those that are interested.

## Usage
The model can be trained using `train_model.py`:

```
python3 train_model.py
```

with the model being saved as a checkpoint after each epoch in the `models` directory. You can then generate text with the model using `generation.py`:

```python
python3 generation.py
```

where you'll be able to provide your own prompt to the model. You can also train your own tokeniser if you so wish:

```python
python3 tokeniser.py
```

## Sample Output
The model can generate text given a prompt. Example outputs:
```
Prompt: 'Alice was'
Response: "was more than alice could bear she got up in great, and walked off;  the dormouse fell asleep instantly, and neither of the others took the least notice of her going..."
```

Requirements
- PyTorch
- Python 3.x
- NumPy

## Implementation Notes

- The model implements causal (unidirectional) attention to prevent looking at future tokens
- Uses learned positional embeddings rather than fixed sinusoidal embeddings
- Includes dropout for regularization
- Supports different generation strategies (temperature scaling, top-k sampling)

## Limitations

- Uses simple word-level tokenization instead of more sophisticated subword tokenization
- Trained on a limited corpus (Alice in Wonderland)
- Relatively small model size compared to state-of-the-art GPT variants
- The model is not fine-tuned for a specific task, such as question and answering

## References
Based on the GPT architecture described in "Improving Language Understanding by Generative Pre-Training" by Radford et al. (2018).

This implementation is intended for educational purposes to understand the core concepts behind transformer-based language models. For production use cases, consider using established libraries and pre-trained models.