from transformers import SpeechEncoderDecoderModel, Wav2Vec2Processor
from datasets import load_dataset
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
# Inference: Translate English speech to German
generated = model.generate(input_values)
decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
decoded
'Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.'

# Training: Train model on English transcription
labels = processor(text=ds[0]["text"], return_tensors="pt").input_ids

loss = model(input_values, labels=labels).loss
loss.backward()
