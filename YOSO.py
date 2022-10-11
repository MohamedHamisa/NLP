import torch
from transformers import AutoTokenizer, YosoForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("uw-madison/yoso-4096")
model = YosoForSequenceClassification.from_pretrained("uw-madison/yoso-4096")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = YosoForSequenceClassification.from_pretrained("uw-madison/yoso-4096", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
