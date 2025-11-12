from collections import Counter
import json
import torch
from model import LSTMClassifier
import tkinter as tk
from tkinter import ttk


VOCAB_PATH = f"./results/vocab.json"
MODEL_PATH = f"./results/lstm_classifier50_64_rus.pth"

CATEGORIES = {
    0: 'Мировые новости',
    1: 'Спорт',
    2: 'Бизнес и экономика',
    3: 'Наука и техника'
}


def tokenize(text):
    return text.lower().split()


def text_to_ids(text, vocab):
    return [vocab.get(tok, 1) for tok in tokenize(text)]


def load_model():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    model = torch.load(MODEL_PATH, weights_only=False, map_location='cuda')
    model.to('cuda')
    model.eval()
    return model, vocab


def predict(input_dict, model, vocab):
    inp = torch.tensor(text_to_ids(input_dict['title'] + " " + input_dict['description'], vocab), dtype=torch.long).cuda()
    with torch.no_grad():
        output = model(inp)
    
    pred = torch.argmax(output)
    return CATEGORIES[pred.item()]


class NewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор новостей")
        self.root.geometry("600x500")
        
        # Загружаем модель при инициализации
        self.model, self.vocab = load_model()
        
        # Заголовок
        title_label = tk.Label(root, text="Заголовок:", font=('Arial', 12))
        title_label.pack(pady=(20, 5))
        
        self.title_entry = tk.Entry(root, width=60, font=('Arial', 11))
        self.title_entry.pack(pady=5)
        
        # Описание
        desc_label = tk.Label(root, text="Описание:", font=('Arial', 12))
        desc_label.pack(pady=(20, 5))
        
        self.desc_text = tk.Text(root, width=60, height=10, font=('Arial', 11))
        self.desc_text.pack(pady=5)
        
        # Кнопка
        self.classify_button = tk.Button(
            root, 
            text="Определить тип", 
            command=self.classify_news,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        )
        self.classify_button.pack(pady=20)
        
        # Результат
        self.result_label = tk.Label(root, text="", font=('Arial', 12, 'bold'), fg='blue')
        self.result_label.pack(pady=10)
    
    def classify_news(self):
        # Получаем данные из полей
        title = self.title_entry.get()
        description = self.desc_text.get("1.0", tk.END).strip()
        
        if not title or not description:
            self.result_label.config(text="Заполните оба поля!", fg='red')
            return
        
        # Формируем входные данные
        input_dict = {
            'title': title,
            'description': description
        }
        
        # Получаем предсказание
        try:
            result = predict(input_dict, self.model, self.vocab)
            self.result_label.config(text=f"Предсказанный тип: {result}", fg='blue')
        except Exception as e:
            self.result_label.config(text=f"Ошибка: {str(e)}", fg='red')


if __name__ == "__main__":
    root = tk.Tk()
    app = NewsClassifierApp(root)
    root.mainloop()
