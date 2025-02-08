# Simple Language Models Implementation

This repository contains implementations of two different language models: a character-level LSTM model and a Transformer-based model. Both models are trained to generate text based on the provided training data.

## Project Structure

.
├── simple_llm.py        # LSTM-based language model
├── transformer_llm.py   # Transformer-based language model
├── training_data.txt    # Training data for the models
└── README.md           # This file
```

## Features

- Two different architecture implementations:
  - Character-level LSTM model with dropout and learning rate scheduling
  - Transformer model with positional encoding and self-attention
- Temperature-controlled text generation
- Configurable model parameters
- Learning rate scheduling for optimal training
- Dropout for regularization
- Customizable training data

## Requirements

```bash
pip install torch torchvision torchaudio
```

## Usage

1. LSTM Model:
```bash
python simple_llm.py
```

2. Transformer Model:
```bash
python transformer_llm.py
```

Both models will:
1. Load training data from `training_data.txt`
2. Train on the text data
3. Generate sample text with different temperature settings

## Model Parameters

### LSTM Model
- Hidden size: 512
- Number of layers: 4
- Dropout: 0.2
- Learning rate: 0.002
- Sequence length: 50
- Training epochs: 500

### Transformer Model
- Model dimension: 256
- Number of heads: 8
- Number of layers: 4
- Dropout: 0.1
- Learning rate: 0.001
- Sequence length: 64
- Training epochs: 200

## Customization

You can modify the training data by editing `training_data.txt`. The models will automatically train on the new content.

To adjust model parameters, modify the respective variables in each model's script:
- For LSTM: Edit parameters in `simple_llm.py`
- For Transformer: Edit parameters in `transformer_llm.py`

## Text Generation

Both models support temperature-controlled text generation:
- Lower temperature (e.g., 0.5): More focused, conservative text
- Higher temperature (e.g., 1.0): More creative, diverse text

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Acknowledgments

This implementation is inspired by various language model architectures and is intended for educational purposes.