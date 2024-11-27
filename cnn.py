import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Função para preprocessamento
def preprocess_data(df, normalize=False, encode_quarter=False):
    # Remove coluna com variável de saída
    df_features = df.drop(columns=["Consumption"])

    # Codificar `Quarter`
    if encode_quarter:
        df_features['Quarter'] = df_features['Quarter'].str[-2:]
        encoder = OneHotEncoder()
        quarters_encoded = encoder.fit_transform(df_features[["Quarter"]])
        quarters_encoded_df = pd.DataFrame(
            quarters_encoded.toarray(), columns=encoder.get_feature_names_out(["Quarter"])
        ).set_index(df_features.index)
        df_features = pd.concat([df_features, quarters_encoded_df], axis=1)
    
    df_features.drop(columns=["Quarter"], inplace=True)

    # Normalizar se necessário
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(df_features)
        df_features = pd.DataFrame(features, columns=df_features.columns)

    X = df_features.values
    y = df["Consumption"].values
    return X, y

# Modelo de convolução 1D
class Conv1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hidden_channels, n_layers):
        super(Conv1DModel, self).__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"conv{i}", nn.Conv1d(
                in_channels=1 if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding="same",
            ))
            setattr(self, f"relu{i}", nn.ReLU())
        self.fc = nn.Linear(hidden_channels * input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adiciona o canal para 1D CNN
        for i in range(self.n_layers):
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"relu{i}")(x)

        x = x.view(x.size(0), -1) # Flatten para camada totalmente conectada
        x = self.fc(x)
        return x
    
# Treinamento do modelo
def train_model(X_train, y_train, X_val, y_val, input_dim, kernel_size, hidden_channels, n_layers, lr, epochs):
    model = Conv1DModel(input_dim, 1, kernel_size, hidden_channels, n_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions.squeeze(), y_train)
        loss.backward()
        optimizer.step()

    # Avaliar no conjunto de validação
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val).squeeze()
        mse = mean_squared_error(y_val, val_predictions)

    return model, mse


# Grid search para encontrar melhores hiperparâmetros
def grid_search(X, y, kernel_sizes, hidden_channels, num_layers,lr, epochs, num_runs=5):
    results = []
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Converter para tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    for layers in num_layers:
        for kernel_size in kernel_sizes:
            row = []
            for hidden_channel in hidden_channels:
                mse_values = []
                for _ in range(num_runs):
                    _, mse = train_model(X_train, y_train, X_val, y_val, X_train.shape[1], kernel_size, hidden_channel, layers, lr, epochs)
                    mse_values.append(mse)

                mse_mean = np.mean(mse_values)
                mse_std = np.std(mse_values)

                row.append((kernel_size, hidden_channel, layers, mse_mean, mse_std))
            results.extend(row)

    # Criar DataFrame para resultados
    results_df = pd.DataFrame(results, columns=["Kernel Size", "Hidden Channels", "Num Layers", "MSE Mean", "MSE Std"])
    return results_df