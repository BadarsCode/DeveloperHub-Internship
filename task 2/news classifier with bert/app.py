import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():
    model_path = "badarscode/news-bert-model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("📰 News Topic Classifier using BERT")
st.write("Enter a news headline and the model will predict its category.")

text = st.text_input("News Headline")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        st.subheader("Prediction Probabilities")
        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[i]*100:.2f}%")

        pred_class = torch.argmax(probs).item()
        st.success(f"Predicted Category: **{labels[pred_class]}**")
