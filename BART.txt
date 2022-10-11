import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()  # disable dropout (or leave in train mode to finetune)

# Download bart.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz

# Load the model in fairseq
from fairseq.models.bart import BARTModel
bart = BARTModel.from_pretrained('/path/to/bart.large', checkpoint_file='model.pt')
bart.eval()  # disable dropout (or leave in train mode to finetune)

tokens = bart.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
bart.decode(tokens)  # 'Hello world!'

# Extract the last layer's features
last_layer_features = bart.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features from decoder (layer 0 is the embedding layer)
all_layers = bart.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)

# Download BART already finetuned for MNLI
bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = bart.encode('BART is a seq2seq model.', 'BART is not sequence to sequence.')
bart.predict('mnli', tokens).argmax()  # 0: contradiction
#expects the input sentences to be formatted as a list of tokens

# Encode another pair of sentences
tokens = bart.encode('BART is denoising autoencoder.', 'BART is version of autoencoder.')
bart.predict('mnli', tokens).argmax()  # 2: entailment

#Register a new (randomly initialized) classification head
bart.register_classification_head('new_task', num_classes=3)
logprobs = bart.predict('new_task', tokens)

import torch
from fairseq.data.data_utils import collate_tokens

bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()

batch_of_pairs = [
    ['BART is a seq2seq model.', 'BART is not sequence to sequence.'],
    ['BART is denoising autoencoder.', 'BART is version of autoencoder.'],
]

batch = collate_tokens(
    [bart.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = bart.predict('mnli', batch)
print(logprobs.argmax(dim=1))
# tensor([0, 2])

# Using the GPU:
bart.cuda()
bart.predict('new_task', tokens)

bart = torch.hub.load('pytorch/fairseq', 'bart.base')
bart.eval()
bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10)

bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10, match_source_len=False)

bart.cuda()
bart.fill_mask(['The cat <mask> on the <mask>.', 'The dog <mask> on the <mask>.'], topk=3, beam=10)

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))

