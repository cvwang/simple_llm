import torch
import torch.nn as nn
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=1024, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_length]
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded, src_mask)
        output = self.output_layer(output)
        return output

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def main():
    # Load text from file
    text_path = "training_data.txt"
    if not os.path.exists(text_path):
        with open(text_path, 'w') as f:
            f.write("""The art of programming is the skill of controlling complexity.
            Programming languages are tools for creating software.
            Artificial intelligence and machine learning are transforming technology.""")
    
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Model parameters
    d_model = 256
    nhead = 8
    num_layers = 4
    dropout = 0.1
    
    # Create model
    model = TransformerLLM(vocab_size, d_model, nhead, num_layers, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 200
    sequence_length = 64
    batch_size = 32

    # Convert text to indices
    text_indices = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Generate random starting points for sequences
        for i in range(0, len(text_indices) - sequence_length, batch_size):
            # Prepare batch
            input_seq = text_indices[i:i + sequence_length].unsqueeze(0)
            target_seq = text_indices[i + 1:i + sequence_length + 1].unsqueeze(0)
            
            # Create mask for self-attention
            src_mask = generate_square_subsequent_mask(sequence_length)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_seq, src_mask)
            
            # Calculate loss
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
    
    # Text generation function
    def generate_text(model, start_text, length=100, temperature=1.0):
        model.eval()
        current_indices = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0)
        generated_text = start_text
        
        with torch.no_grad():
            for _ in range(length):
                mask = generate_square_subsequent_mask(len(current_indices[0]))
                output = model(current_indices, mask)
                
                # Get next character probabilities
                next_char_logits = output[0, -1] / temperature
                next_char_probs = torch.softmax(next_char_logits, dim=-1)
                next_char_idx = torch.multinomial(next_char_probs, 1).item()
                
                # Add to generated text
                generated_text += idx_to_char[next_char_idx]
                current_indices = torch.cat([current_indices, 
                                          torch.tensor([[next_char_idx]], dtype=torch.long)], dim=1)
                
                # Keep sequence length manageable
                if current_indices.size(1) > sequence_length:
                    current_indices = current_indices[:, -sequence_length:]
        
        return generated_text

    # Generate samples
    print("\nGenerated text (temperature=0.7):")
    print(generate_text(model, "The ", length=200, temperature=0.7))
    
    print("\nGenerated text (temperature=1.0):")
    print(generate_text(model, "The ", length=200, temperature=1.0))

if __name__ == "__main__":
    main() 