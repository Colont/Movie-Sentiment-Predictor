import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # batch_first for batch-major inputs
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Embedding: Convert token IDs to embeddings
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # RNN: Process the embeddings
        output, hidden = self.rnn(embedded)  # output: [batch_size, seq_len, hidden_dim]
                                            # hidden: [1, batch_size, hidden_dim]
        
        # Final layer: Use the hidden state of the last layer
        hidden = hidden.squeeze(0)          # hidden: [batch_size, hidden_dim]
        return self.fc(hidden)              # [batch_size, output_dim]


# Test code (outside the class)
batch_size = 32
seq_len = 50
vocab_size = 1000
embedding_dim = 100
hidden_dim = 128
output_dim = 2

# Instantiate the model
model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)

# Dummy input (batch of tokenized sequences)
input_data = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]

# Forward pass
output = model(input_data)  # [batch_size, output_dim]

print(output.shape)  # Should print: torch.Size([32, 2])
