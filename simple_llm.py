import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple character-level language model
class SimpleCharLLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleCharLLM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                 weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

# Example usage
def main():
    # Larger, more structured training text
    text = """
    The quick brown fox jumps over the lazy dog. 
    Machine learning is a fascinating field of study.
    Neural networks can learn patterns in data.
    Language models help us generate human-like text.
    Artificial intelligence continues to evolve rapidly.
    Deep learning has revolutionized natural language processing.
    """
    
    # Create character to index mapping
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Increased model parameters
    input_size = len(chars)
    hidden_size = 256  # Increased from 128
    num_layers = 3    # Increased from 2
    output_size = len(chars)
    
    # Create model
    model = SimpleCharLLM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert text to one-hot encoded tensors
    def text_to_tensor(text, char_to_idx):
        tensor = torch.zeros(len(text), 1, len(char_to_idx))
        for i, char in enumerate(text):
            tensor[i][0][char_to_idx[char]] = 1
        return tensor
    
    # Increased training epochs
    num_epochs = 500  # Increased from 100
    
    # Modified training loop with sequence batching
    sequence_length = 50  # Increased context window
    
    for epoch in range(num_epochs):
        model.zero_grad()
        total_loss = 0
        
        # Create training sequences
        for i in range(0, len(text) - sequence_length, sequence_length):
            input_seq = text_to_tensor(text[i:i + sequence_length], char_to_idx)
            target_seq = torch.LongTensor([char_to_idx[char] for char in text[i + 1:i + sequence_length + 1]])
            
            output, hidden = model(input_seq)
            loss = criterion(output.view(-1, len(chars)), target_seq)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:  # Changed from 10
            print(f'Epoch {epoch}, Average Loss: {total_loss:.4f}')
    
    # Generate some text
    def generate_text(model, start_char, char_to_idx, idx_to_char, length=100, temperature=0.8):
        model.eval()
        current_char = start_char
        generated_text = current_char
        hidden = None
        
        for _ in range(length):
            input_tensor = text_to_tensor(current_char, char_to_idx)
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature scaling
            output = output[-1].div(temperature)
            probs = torch.softmax(output, dim=1)
            
            # Sample with higher probability for more likely characters
            char_idx = torch.multinomial(probs, 1).item()
            current_char = idx_to_char[char_idx]
            generated_text += current_char
        
        return generated_text

    # Generate multiple samples with different temperatures
    print("\nGenerated text (temperature=0.5, more focused):")
    print(generate_text(model, 'T', char_to_idx, idx_to_char, temperature=0.5))
    
    print("\nGenerated text (temperature=1.0, more creative):")
    print(generate_text(model, 'T', char_to_idx, idx_to_char, temperature=1.0))

if __name__ == "__main__":
    main() 