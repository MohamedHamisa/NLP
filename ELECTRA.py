!pip install tensorflow==1.15
!pip install transformers==2.8.0
!git clone https://github.com/google-research/electra.git
import os
import json
from transformers import AutoTokenizer

DATA_DIR = "./data" #@param {type: "string"}
TRAIN_SIZE = 1000000 #@param {type:"integer"}
MODEL_NAME = "electra-spanish" #@param {type: "string"}
# Download and unzip the Spanish movie substitle dataset
if not os.path.exists(DATA_DIR):
  !mkdir -p $DATA_DIR
  !wget "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/es.txt.gz" -O $DATA_DIR/OpenSubtitles.txt.gz
  !gzip -d $DATA_DIR/OpenSubtitles.txt.gz
  !head -n $TRAIN_SIZE $DATA_DIR/OpenSubtitles.txt > $DATA_DIR/train_data.txt 
  !rm $DATA_DIR/OpenSubtitles.txt


# Save the pretrained WordPiece tokenizer to get vocab.txt
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.save_pretrained(DATA_DIR)
#We use build_pretraining_dataset.py to create a pre-training dataset from a dump of raw text.

!python3 electra/build_pretraining_dataset.py \
  --corpus-dir $DATA_DIR \
  --vocab-file $DATA_DIR/vocab.txt \
  --output-dir $DATA_DIR/pretrain_tfrecords \
  --max-seq-length 128 \
  --blanks-separate-docs False \
  --no-lower-case \
  --num-processes 5

#TRAINING
hparams = {
    "do_train": "true",
    "do_eval": "false",
    "model_size": "small",
    "do_lower_case": "false",
    "vocab_size": 119547,
    "num_train_steps": 100,
    "save_checkpoints_steps": 100,
    "train_batch_size": 32,
}

with open("hparams.json", "w") as f:
    json.dump(hparams, f)

!python3 electra/run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name $MODEL_NAME \
  --hparams "hparams.json"

#If you are training on a virtual machine, run the following lines on the terminal to moniter the training process with TensorBoard.

pip install -U tensorboard
tensorboard dev upload --logdir data/models/electra-spanish

!git clone https://github.com/lonePatient/electra_pytorch.git #to convert Tensorflow checkpoints to PyTorch using huggingface
MODEL_DIR = "data/models/electra-spanish/"

config = {
  "vocab_size": 119547,
  "embedding_size": 128,
  "hidden_size": 256,
  "num_hidden_layers": 12,
  "num_attention_heads": 4,
  "intermediate_size": 1024,
  "generator_size":"0.25",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "max_position_embeddings": 512,
  "type_vocab_size": 2,
  "initializer_range": 0.02
}

with open(MODEL_DIR + "config.json", "w") as f:
    json.dump(config, f)
!python electra_pytorch/convert_electra_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=$MODEL_DIR \
    --electra_config_file=$MODEL_DIR/config.json \
    --pytorch_dump_path=$MODEL_DIR/pytorch_model.bin

#use ELECTRA
import torch
from transformers import ElectraForPreTraining, ElectraTokenizerFast

discriminator = ElectraForPreTraining.from_pretrained(MODEL_DIR)
tokenizer = ElectraTokenizerFast.from_pretrained(DATA_DIR, do_lower_case=False)
sentence = "Los pájaros están cantando" # The birds are singing
fake_sentence = "Los pájaros están hablando" # The birds are speaking 

fake_tokens = tokenizer.tokenize(fake_sentence, add_special_tokens=True)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
predictions = discriminator_outputs[0] > 0

[print("%7s" % token, end="") for token in fake_tokens]
print("\n")
[print("%7s" % int(prediction), end="") for prediction in predictions.tolist()];
