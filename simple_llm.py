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
    # Sample text for training
    text = """
    Hello, this is a simple example text.
    We're training a tiny language model.
    It will learn to predict characters!
    """
    
    # Create character to index mapping
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Model parameters
    input_size = len(chars)
    hidden_size = 128
    num_layers = 2
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
    
    # Training loop
    num_epochs = 100
    sequence_length = 25
    
    for epoch in range(num_epochs):
        model.zero_grad()
        loss = 0
        
        input_seq = text_to_tensor(text[:-1], char_to_idx)
        target_seq = torch.LongTensor([char_to_idx[char] for char in text[1:]])
        
        output, hidden = model(input_seq)
        loss = criterion(output.view(-1, len(chars)), target_seq)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Generate some text
    def generate_text(model, start_char, char_to_idx, idx_to_char, length=100):
        model.eval()
        current_char = start_char
        generated_text = current_char
        hidden = None
        
        for _ in range(length):
            input_tensor = text_to_tensor(current_char, char_to_idx)
            output, hidden = model(input_tensor, hidden)
            
            # Sample from the output distribution
            probs = torch.softmax(output[-1], dim=1)
            char_idx = torch.multinomial(probs, 1).item()
            current_char = idx_to_char[char_idx]
            generated_text += current_char
        
        return generated_text

    # Generate sample text
    print("\nGenerated text:")
    print(generate_text(model, 'H', char_to_idx, idx_to_char))

if __name__ == "__main__":
    main() 