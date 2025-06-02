import torch
import torch.nn as nn

HIDDEN_SIZE_LSTM = 64
HIDDEN_SIZE_FC = 32


class SalesPredictorLSTM(nn.Module):
    def __init__(
        self,
        num_dynamic_features,
        lstm_hidden_size,
        fc_hidden_size,
        num_static_features,
    ):
        super().__init__()
        self.num_static_features = num_static_features

        self.static_dim = 16

        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.static_dim),
            nn.ReLU(),
        )

        # LSTM for dynamic features
        self.lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=1,
        )

        self.fc1 = nn.Linear(lstm_hidden_size + self.static_dim, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)

    def forward(self, dynamic_seq, static_feats):
        # dynamic_seq: (batch, seq_len, num_dynamic_features)
        # static_feats: (batch, num_static_features)

        # LSTM part
        lstm_out, (h_n, c_n) = self.lstm(dynamic_seq)
        lstm_last_hidden = h_n[-1]

        # Static part
        static_features = self.static_mlp(static_feats)

        # Combine LSTM output and embeddings
        combined = torch.cat((lstm_last_hidden, static_features), dim=1)

        # Pass through FC layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return self.relu(x)

