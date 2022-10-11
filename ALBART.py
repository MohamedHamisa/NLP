# TODO(lanzhzh): Add support for 2.x.
%tensorflow_version 1.x
import os
import pprint
import json
import tensorflow as tf

#If there is a problem with the connection wie print a error message, else we print the ip address of the tpu instance
assert "COLAB_TPU_ADDR" in os.environ, "ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!"
TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"] 
TPU_TOPOLOGY = "2x2"
print("TPU address is", TPU_ADDRESS)

from google.colab import auth
auth.authenticate_user()
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

  # Upload credentials to TPU.
  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
    # Now credentials are set for all future sessions on this TPU.

#TODO(lanzhzh): Add pip support
import sys

!test -d albert || git clone https://github.com/google-research/albert albert
if not 'albert' in sys.path:
  sys.path += ['albert']
  
!pip install sentencepiece

# Please find the full list of tasks and their fintuning hyperparameters
# here https://github.com/google-research/albert/blob/master/run_glue.sh

BUCKET = "albert_tutorial_glue" #@param { type: "string" }
TASK = 'MRPC' #@param {type:"string"}
# Available pretrained model checkpoints:
#   base, large, xlarge, xxlarge
ALBERT_MODEL = 'base' #@param {type:"string"}

TASK_DATA_DIR = 'glue_data'

BASE_DIR = "gs://" + BUCKET
if not BASE_DIR or BASE_DIR == "gs://":
  raise ValueError("You must enter a BUCKET.")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = 'gs://{}/albert-tfhub/models/{}'.format(BUCKET, TASK)
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# Download glue data.
! test -d download_glue_repo || git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git download_glue_repo
!python download_glue_repo/download_glue_data.py --data_dir=$TASK_DATA_DIR --tasks=$TASK
print('***** Task data directory: {} *****'.format(TASK_DATA_DIR))

ALBERT_MODEL_HUB = 'https://tfhub.dev/google/albert_' + ALBERT_MODEL + '/3'

os.environ['TFHUB_CACHE_DIR'] = OUTPUT_DIR
!python -m albert.run_classifier \
  --data_dir="glue_data/" \
  --output_dir=$OUTPUT_DIR \
  --albert_hub_module_handle=$ALBERT_MODEL_HUB \
  --spm_model_file="from_tf_hub" \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --max_seq_length=512 \
  --optimizer=adamw \
  --task_name=$TASK \
  --warmup_step=200 \
  --learning_rate=2e-5 \
  --train_step=800 \
  --save_checkpoints_steps=100 \
  --train_batch_size=32 \
  --tpu_name=$TPU_ADDRESS \
  --use_tpu=True
