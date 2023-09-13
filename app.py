import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import nltk

# Download NLTK data
nltk.download('punkt')

# Function to process the text with DistilBART and get a single summary
def process_text(input_text):
    checkpoint = "facebook/bart-large-cnn"  # Use a different checkpoint

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Split text into sentences
    sentences = nltk.tokenize.sent_tokenize(input_text)

    # Initialize variables
    length = 0
    chunk = ""

    for sentence in sentences:
        combined_length = len(tokenizer.tokenize(sentence)) + length

        if combined_length <= tokenizer.max_len_single_sentence:
            chunk += sentence + " "
            length = combined_length
        else:
            break  # Stop after the first chunk (single summary)

    # Generate a single summary for the first chunk
    input_data = tokenizer(chunk, return_tensors="tf")  # Use TensorFlow
    output = model.generate(**input_data)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return summary

# Streamlit UI with styling
st.title("Text Summarizer with DistilBART")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    # Original Text
    st.subheader("Original Text")
    st.write(file_contents)

    # Summarized Text
    st.subheader("Summarized Text")

    # Display a single summary
    summary = process_text(file_contents)
    st.info("Summary:")
    st.write(summary)
