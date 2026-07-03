# 安裝所需的套件
#!pip install transformers datasets scikit-learn

# 載入相關模組
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from keras.optimizers import Adam
# 設定資料列
DATA_COLUMN = 'sha256'
LABEL_COLUMN = 'label'

# 讀取資料集
df_train = pd.read_json("C:/Bert.dataset/ember_dataset (1)/ember/train_features_0.jsonl", lines=True)
df_test = pd.read_json("C:/Bert.dataset\ember_dataset (1)/ember/test_features.jsonl", lines=True)
df_fake = pd.read_json("C:/Bert.dataset\ember_dataset (1)/ember/train_features_1.jsonl", lines=True)

# 整合資料並清理
train_df = pd.concat([df_fake, df_train])
train_df = train_df[train_df[LABEL_COLUMN] != -1]  # 移除無效標籤
train_df = train_df.sample(frac=1).reset_index(drop=True)  # 打亂資料

# 分割資料集
train_data, val_data = train_test_split(train_df, test_size=0.2)

# 初始化 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 定義資料轉換函數
def convert_to_bert_input(data, tokenizer, max_length=128):
    # 使用 tokenizer 處理文本
    features = tokenizer(
        list(data[DATA_COLUMN]),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    
    # 創建輸入字典
    train_dataset = {
        "input_ids": features["input_ids"],
        "attention_mask": features["attention_mask"],
        "token_type_ids": features["token_type_ids"]
    }
    
    # 轉換標籤為 tensorflow tensor
    labels = tf.convert_to_tensor(data[LABEL_COLUMN].tolist())
    
    return train_dataset, labels

# 準備訓練與驗證資料
train_features, train_labels = convert_to_bert_input(train_data, tokenizer)
val_features, val_labels = convert_to_bert_input(val_data, tokenizer)


# 建立 BERT 模型
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 編譯模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# 訓練模型
history = model.fit(
    train_features,
    train_labels,
    validation_data=(val_features, val_labels),
    epochs=3,
    batch_size=16  # 減少批量大小以降低記憶體使用
)

# 評估模型
results = model.evaluate(val_features, val_labels)
print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")
