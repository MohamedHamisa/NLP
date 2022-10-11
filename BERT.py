import tensorflow_hub as hub
import tensorflow_text as text
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" #to preprocess the text
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"  #model

bert_preprocess_model = hub.KerasLayer(preprocess_url) # to stay organized with collections save and categorize the content based on your preferences  
text_test = ['nice movie indeed','I love python programming']
text_preprocessed = bert_preprocess_model(text_test)
text_preprocessed.keys()  # to describe the data

text_preprocessed['input_mask']
#for 1st sentence the mask will be 5 because there is a CLS speacial token at the beginning and between each word there is a seperate token

text_preprocessed['input_type_ids']

text_preprocessed['input_word_ids']

bert_model = hub.KerasLayer(encoder_url) 
bert_results = bert_model(text_preprocessed) #like a function pointer
bert_results.keys() 

bert_results['sequence_output']  # individual or for each word embedding for the entire sentences

bert_results['pooled_output'] #embedding for the entire sentences

len(bert_results['encoder_outputs'])

bert_results['encoder_outputs'][-1] == bert_results['sequence_output']  #(2 sen, 128word includin Padding, 768 embedding)

bert_results['encoder_outputs']

