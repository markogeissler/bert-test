import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    model = BertForSequenceClassification.from_pretrained("younoger/YGBNumbersBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Nachricht des Agents')
button = st.button("Auswerten")

d = {
    
  1:'Nummer',
  0:'Keine Nummer'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])