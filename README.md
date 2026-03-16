Amazon Review Rating Classification – BERT
Fine-tuning of bert-base-uncased for 5-class sentiment classification on Amazon product reviews. Trained on 500,000 balanced reviews across 5 product categories.
Task
Predict the star rating (1–5) of an Amazon review based on title and review text.
Dataset
CategorySourceVideo GamesMcAuley-Lab/Amazon-Reviews-2023BooksMcAuley-Lab/Amazon-Reviews-2023Digital MusicMcAuley-Lab/Amazon-Reviews-2023Industrial & ScientificMcAuley-Lab/Amazon-Reviews-2023Movies & TVMcAuley-Lab/Amazon-Reviews-2023
100,000 reviews per category → balanced to 40,000 per rating class (under-/oversampling).
Model

Base: bert-base-uncased
Task head: BertForSequenceClassification (5 labels)
Optimizer: AdamW (lr=1e-5, weight_decay=0.03)
Dropout: 0.2
Max sequence length: 128 tokens
Batch size: 64
Epochs: 10
Device: CUDA

Output
OutputDescriptionTraining/Validation accuracy curvesPer-epoch accuracy plotLoss curveCross-entropy loss per epochConfusion matrix (normalized)Heatmap over all 5 rating classesClassification reportPrecision, Recall, F1 per class
Requirements
bashpip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn
Usage
bashpython main.py
```

Requires GPU (CUDA). Dataset is streamed directly from HuggingFace Hub — no manual download needed.

## Pipeline
```
HuggingFace Streaming
    └── 5 categories × 100k reviews
        └── Concat + balance (40k per class)
            └── BERT Tokenizer (max_length=128)
                └── TensorDataset → DataLoader (80/20 split)
                    └── BertForSequenceClassification
                        └── AdamW → 10 Epochs
                            └── Metrics & Visualizations
