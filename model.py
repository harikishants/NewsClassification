import torch
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_heads=4, num_layers=2, num_classes=5, max_len = 512):
        super().__init__()

        self.hparams = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'max_len': max_len,
        }

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(input_ids.device)

        token_embed = self.token_embedding(input_ids)
        pos_embed = self.position_embedding(pos_ids)
        x = token_embed + pos_embed

        attn_mask = attention_mask == 0 # pytorch expects a mask of true where padding is present
        encoded = self.encoder(x, src_key_padding_mask = attn_mask)
        cls_rep = encoded[:, 0, :] # classfication representation token, shape: (batch_size, seq_len, embed_dim)

        output = self.classifier(cls_rep)

        return output

