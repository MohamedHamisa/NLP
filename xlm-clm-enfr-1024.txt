import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel

tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")

print(tokenizer.lang2id) # language used and are identified by the tokenizer’s lang2id and id2lang attributes.
{'en': 0, 'fr': 1}

input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1


language_id = tokenizer.lang2id["en"]  # 0
langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

# We reshape it to be of size (batch_size, sequence_length)
langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)

outputs = model(input_ids, langs=langs)
