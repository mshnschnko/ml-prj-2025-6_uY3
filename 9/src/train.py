# ========== Импорты ==========
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from model import LSTMClassifier

# ========== 1. Настройки ==========
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
EPOCHS = 50
LR = 1e-4
MAX_LEN = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = f"lstm_classifier_weights{EPOCHS}_{BATCH_SIZE}_rus.pth"
FIGURE_PATH = f"training_curves{EPOCHS}_{BATCH_SIZE}_rus.png"
MODEL_PATH = f"lstm_classifier{EPOCHS}_{BATCH_SIZE}_rus.pth"
VOCAB_PATH = f"./results/vocab.json"

# ========== 2. Загрузка данных ==========
train_df = pd.read_csv("./dataset/train_translated.csv")
test_df = pd.read_csv("./dataset/test_translated.csv")

def combine_text(df):
    df["text"] = (df["Title"].fillna("") + " " + df["Description"].fillna("")).str.strip()
    return df

train_df = combine_text(train_df)
test_df = combine_text(test_df)

# Метки
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["Class Index"])
test_df["label"] = label_encoder.transform(test_df["Class Index"])

# ========== 3. Токенизация ==========
def tokenize(text):
    return text.lower().split()

# Строим словарь
counter = Counter()
for text in train_df["text"]:
    counter.update(tokenize(text))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(10000):
    vocab[word] = len(vocab)

with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

def text_to_ids(text):
    return [vocab.get(tok, 1) for tok in tokenize(text)]

# ========== 4. Dataset и DataLoader ==========
class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = [torch.tensor(text_to_ids(t), dtype=torch.long) for t in df["text"]]
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    padded = padded[:, :MAX_LEN]
    return padded, torch.stack(labels)

train_ds = TextDataset(train_df)
test_ds = TextDataset(test_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

num_classes = len(label_encoder.classes_)
model = LSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========== 6. Обучение ==========
train_losses, test_losses, train_accs, test_accs = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)

    # === Валидация ===
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [test]"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    test_losses.append(total_loss / len(test_loader))
    test_accs.append(correct / total)

    print(f"Epoch {epoch+1}: "
          f"Train loss={train_losses[-1]:.4f}, acc={train_accs[-1]:.4f}; "
          f"Test loss={test_losses[-1]:.4f}, acc={test_accs[-1]:.4f}")

# ========== 7. Графики ==========
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
plt.show()

# ========== 8. Сохранение модели ==========
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"✅ Веса сохранены в {WEIGHTS_PATH}")
torch.save(model, MODEL_PATH)
print(f"✅ Модель сохранена в {MODEL_PATH}")
