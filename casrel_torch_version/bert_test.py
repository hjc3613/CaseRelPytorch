from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import torch
import jieba

model = DistilBertModel.from_pretrained(r'D:\bert-models\distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained(r'D:\bert-models\distilbert-base-uncased')
sentence = '我爱你[PAD][PAD][PAD]'
ids = tokenizer.encode(sentence)
input_tensor = torch.tensor([ids])
result = model(input_tensor)
print(result[0].shape)