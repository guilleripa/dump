import torch
import torch.nn as nn

# Obtener la matriz de embeddings desde el diccionario     ---> FREEZARLO (PARA NO CATASTROFIC FORGETING)
# embeddings_matrix = torch.tensor(list(embeddings_index.values()))


# Define una clase para el modelo
class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
    ):
        self.bidirectional = bidirectional
        super(RNNModel, self).__init__()

        # Capa de embedding con los embeddings pre-entrenados (la congele)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa RNN (LSTM en este caso)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        # Capa lineal de salida
        self.fc = nn.Linear(
            hidden_dim * 2 if self.bidirectional else hidden_dim, output_dim
        )

        # Capa de dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )
        _, (hidden, _) = self.rnn(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(
        #     packed_output, batch_first=True
        # )
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            if self.bidirectional
            else hidden[-1, :, :]
        )
        output = self.fc(hidden)
        return output
