import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt 

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return torch.sigmoid(output)


# Padding
# Ku kratším sekvenciám sa pridajú ďalšie riadky núl, takže všetky majú rovnaký počet riadkov
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])

    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = np.zeros((max_len - len(seq), seq.shape[1]))
            padded_seq = np.vstack((seq, padding))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)

    return np.stack(padded_sequences)


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/S4IKEz/stuff/main/questions1.csv"
    data = pd.read_csv(url, usecols=['question1', 'question2', 'is_duplicate'])

    data['question1'] = data['question1'].str.lower().str.split()
    data['question2'] = data['question2'].str.lower().str.split()

    # Trénujte model Word2Vec
    sentences = data['question1'].tolist() + data['question2'].tolist()
    model_w2v = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=4)

    # Encode questions using Word2Vec
    def encode_questions(question):
        return np.array([model_w2v.wv[word] for word in question])

    data['q1_encoded'] = data['question1'].apply(encode_questions)
    data['q2_encoded'] = data['question2'].apply(encode_questions)

    X1 = pad_sequences(data['q1_encoded'].tolist())
    X2 = pad_sequences(data['q2_encoded'].tolist())
    y = data['is_duplicate'].values

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

    # Parametre
    input_size = 128
    hidden_size = 128
    num_layers = 1
    num_epochs = 15
    learning_rate = 0.001
    batch_size = 32

    # Nevyhnutná konverzia údajov na tenzor

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    X1_train_tensor = torch.tensor(X1_train, dtype=torch.float32).to(device)
    X2_train_tensor = torch.tensor(X2_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_dataset = torch.utils.data.TensorDataset(X1_train_tensor, X2_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Trenovanie
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for i, (q1, q2, labels) in enumerate(train_loader):
            q1_out = model(q1).to(device)
            q2_out = model(q2).to(device)
            #print(q1_out, q2_out);
            out = torch.abs(q1_out - q2_out)

            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item()}")



    # Vysledky
    model.eval()
    with torch.no_grad():
        X1_test_tensor = torch.tensor(X1_test, dtype=torch.float32).to(device)
        X2_test_tensor = torch.tensor(X2_test, dtype=torch.float32).to(device)
        q1_test_out = model(X1_test_tensor)
        q2_test_out = model(X2_test_tensor)
        test_out = torch.abs(q1_test_out - q2_test_out)
        test_preds = (test_out >= 0.379999).type(torch.float32).view(-1)
        y_true = torch.tensor(y_test, dtype=torch.float32).view(-1)
        y_pred = test_preds.cpu()
        cm = confusion_matrix(y_true, y_pred)
        print("Confuzna matica:")
        sns.heatmap(cm,annot=True)
        plt.show()
        
        test_acc = torch.mean((test_preds == torch.tensor(y_test, dtype=torch.float32).to(device)).type(torch.float32)).item()
        print(f"Presnost: {test_acc}")