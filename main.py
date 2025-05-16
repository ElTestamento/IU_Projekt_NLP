from datasets import load_dataset
import pandas as pd

from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

'''
Was der CODE tut:
Daten laden und einen Teil in ein Pandas-DF übergeben und zusammenführen ✓
Tokenisierung ✓
Dataset und DataLoader ✓
BERT-Modell ✓
Optimizer ✓
Device-Konfiguration ✓
Trainings-Schleife ✓
Einfache Metrik ✓
Metriken zur Datenevaluation ✓
Metrik zur Trainingsvisualisierung ✓
Metrik zur Modellstärke ✓
'''

#Kategorien aus dem gesamten Datensatz der Reviews laden: Video_Games,Books ,Digital_Music, Industrial_and_Scientific,Movies_and_TV
games_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="raw_review_Video_Games",trust_remote_code=True,streaming=True)
books_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books",trust_remote_code=True,streaming=True)
music_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Digital_Music",trust_remote_code=True,streaming=True)
scientific_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Industrial_and_Scientific",trust_remote_code=True,streaming=True)
movies_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Movies_and_TV",trust_remote_code=True,streaming=True)

# Für Streaming-Datasets: Begrenzung auf 100000 Einträge und Konvertierung zu Pandas
df_games = pd.DataFrame(list(games_data['full'].take(100000)))
df_books = pd.DataFrame(list(books_data['full'].take(100000)))
df_music = pd.DataFrame(list(music_data['full'].take(100000)))
df_sci = pd.DataFrame(list(scientific_data['full'].take(100000)))
df_movies = pd.DataFrame(list(movies_data['full'].take(100000)))

#Zusammenführen von Title und Text für neue Zeile 'combined'
df = pd.concat([df_games,df_books,df_music,df_sci,df_movies], ignore_index=True)
df['combined'] = df['title'] + " " + df['text']
#Durchschnittliche Textlänge:
df['length_text'] = df['text'].apply(lambda x: len(x.split()))
df['length_mean'] = df.groupby('rating')['length_text'].transform('mean').round(0)

#Visualisierung:--------------------
#Balkendiagramm:Anzahl pro rating(unbalanciert)
rating_bar= plt.bar(df['rating'].value_counts().index,df['rating'].value_counts().values)
plt.xlabel('Rating')
plt.ylabel('Anzahl (Ratings)')
plt.title('Anzahl der Bewertungen pro Rating (unbalanciert)')
plt.show()

target_per_class = 40000
balanced_df = pd.DataFrame()

for rating in range(1, 6):
    class_data = df[df['rating'] == rating]
    if len(class_data) > target_per_class:
        # Untersampling
        sampled_data = class_data.sample(target_per_class, random_state=42)
    elif len(class_data) < target_per_class:
        # Übersampling mit Zurücklegen
        sampled_data = class_data.sample(target_per_class, replace=True, random_state=42)
    else:
        sampled_data = class_data
    balanced_df = pd.concat([balanced_df, sampled_data])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)#balanciertes Rating

#Anzahl der Reviews pro Rating(balanciert):
plt.figure(figsize=(10, 6))
balanced_rating_counts = balanced_df['rating'].value_counts().sort_index()
plt.bar(balanced_rating_counts.index, balanced_rating_counts.values)
plt.xlabel('Rating')
plt.ylabel('Anzahl (Rating)')
plt.title('Verteilung der Ratings (balanciert)')
plt.show()

#Text-Length im Mittel pro Rating(balanciert):
mean_lengths = balanced_df.groupby('rating')['length_text'].mean().round(0)
plt.figure(figsize=(10, 6))
plt.bar(mean_lengths.index, mean_lengths.values)
plt.xlabel('Rating')
plt.ylabel('Durchschnittliche Textlänge (in Wörtern)')
plt.title('Mittlere Textlänge pro Rating (balanciert)')
plt.show()

##Labels auf 0 Referenzieren, also von 1-5 zu 0-4
#Definition der Y-Label: In dieser Fragestellung die Rating-Kategorien
Y_labels = torch.tensor(balanced_df['rating'].values - 1, dtype=torch.long)

#Auswahl BERT-Model
BERT_MODEL = 'bert-base-uncased'
#Tokenizing für BERT-Transformer
tokenized = BertTokenizer.from_pretrained(BERT_MODEL)#Standard BERT-Modell
tokenized_data = tokenized.batch_encode_plus(
    balanced_df['combined'].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)#Bert macht keine lemmatisierung, keine Zeichenentfernung, keine Entfernung von Stoppwörtern

# Extrahieren der Tensoren
X_input = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']

# Dataset erstellen
dataset = TensorDataset(X_input, attention_masks, Y_labels)#Eingabe X-Werte und Masks und Y-Werte zur Herstellung Tensor-Dataset

# Einfacher Train-Test-Split (nicht stratifiziert, da bereits balanciert)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size],
                                 generator=torch.Generator().manual_seed(42))#Gibt aufgeteites Dataset zurück

BATCH_SIZE = 64
# DataLoader erstellen
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

#BERT-Model-Implementieren-----------------------------------------
model = BertForSequenceClassification.from_pretrained(BERT_MODEL,
                                      num_labels=5,#5 labels da 5 Ratinglabel
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      hidden_dropout_prob=0.2)

#Optimizier definieren CUDA aktivieren
optimizer = AdamW(model.parameters(), lr=0.00001, eps=0.00000001, weight_decay=0.03)#Definiert Optimizer
device = 'cuda'
model.to(device)
loss_tracker = []
train_accuracy_history = []
val_accuracy_history = []
x_pred = []
y_true = []

#TRAININGSSCHLEIFE--------------------------------------
print('Das Training startet')
EPOCHS = 10
for epochs in range(EPOCHS):
    print(f'Epoche {epochs+1}/{EPOCHS}')

    model.train()
    total_loss=0

    for batch in train_dataloader:#Schleife iteriert durch den train_dataloader: Jede Iteration gibt einen batch zurück
        X_input,attention_masks,Y_labels=batch#ein Batch ist hier definiert als X_input, attention_mask,Y_labels
        X_input,attention_masks,Y_labels=X_input.to(device),attention_masks.to(device),Y_labels.to(device)
        optimizer.zero_grad()
        output = model(input_ids=X_input,attention_mask=attention_masks,labels=Y_labels)
        loss=output.loss
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()

    avg_loss = total_loss / len(train_dataloader)
    loss_tracker.append(avg_loss)
    print(f"Durchschnittlicher Verlust: {avg_loss}")

    #Evaluierung:
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            X_input,attention_masks,Y_labels=batch
            X_input,attention_masks,Y_labels=X_input.to(device),attention_masks.to(device),Y_labels.to(device)
            output = model(input_ids=X_input,attention_mask=attention_masks)
            logits = output.logits
            _, predicted = torch.max(logits, 1)

            x_pred.extend(predicted.cpu().numpy())
            y_true.extend(Y_labels.cpu().numpy())

    #Accuracy:
    model.eval()
    # Training Accuracy
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for batch in train_dataloader:
            X_input, attention_masks, Y_labels = batch
            X_input, attention_masks, Y_labels = X_input.to(device), attention_masks.to(device), Y_labels.to(device)
            output = model(input_ids=X_input, attention_mask=attention_masks)
            logits = output.logits
            _, predicted = torch.max(logits, 1)

            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(Y_labels.cpu().numpy())

        # Validation Accuracy
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            X_input, attention_masks, Y_labels = batch
            X_input, attention_masks, Y_labels = X_input.to(device), attention_masks.to(device), Y_labels.to(device)
            output = model(input_ids=X_input, attention_mask=attention_masks)
            logits = output.logits
            _, predicted = torch.max(logits, 1)

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(Y_labels.cpu().numpy())

    # Berechnen und Speichern der Genauigkeiten für diese Epoche
    train_accuracy = accuracy_score(train_labels, train_preds)
    val_accuracy = accuracy_score(val_labels, val_preds)

    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)

    print(f"Epoche {epochs+1} - Trainingsgenauigkeit: {train_accuracy:.2f}, Validierungsgenauigkeit: {val_accuracy:.2f}")

# Genauigkeit berechnen
accuracy = np.mean(np.array(y_true) == np.array(x_pred))
print(f"Genauigkeit (Accuracy): {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, x_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report erstellen
report = classification_report(val_labels, val_preds, target_names=['1 Stern', '2 Sterne', '3 Sterne', '4 Sterne', '5 Sterne'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)

#Visualisierung Accuracy:Train/Test
plt.plot(range(1,EPOCHS+1), train_accuracy_history, label='Trainingsgenauigkeit')
plt.plot(range(1,EPOCHS+1), val_accuracy_history, label='Validierungsgenauigkeit')
plt.ylim(0, 1)
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.title('Genauigkeit während des Trainings')
plt.legend()
plt.show()

#Visualisierung LOSS-Funktion:
vis_loss_function = plt.plot(range(EPOCHS), loss_tracker)
plt.xlabel('Epoche')
plt.ylabel('Verlust')
plt.title('Verlust während des Trainings')
plt.show()

# Normalisierte Konfusionsmatrix
plt.figure(figsize=(10, 8))
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['1 Stern', '2 Sterne', '3 Sterne', '4 Sterne', '5 Sterne'],
            yticklabels=['1 Stern', '2 Sterne', '3 Sterne', '4 Sterne', '5 Sterne'])
plt.xlabel('Vorhergesagte Bewertung')
plt.ylabel('Tatsächliche Bewertung')
plt.title('Normalisierte Konfusionsmatrix')
plt.show()
