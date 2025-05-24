import os
import re
import spacy
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def mask_pii(text):
    doc = nlp(text)
    masked = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "EMAIL", "PHONE", "GPE"]:
            masked = masked.replace(ent.text, f"[MASKED_{ent.label_}]")
    masked = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[MASKED_EMAIL]', masked)
    masked = re.sub(r'\b\d{10}\b', '[MASKED_PHONE]', masked)
    return masked

def semantic_chunk(text, max_tokens=200):
    doc = nlp(text)
    chunks = []
    chunk = ""
    for sent in doc.sents:
        if len(chunk) + len(sent.text) <= max_tokens:
            chunk += sent.text + " "
        else:
            chunks.append(chunk.strip())
            chunk = sent.text + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Streamlit UI
st.set_page_config(page_title="Local RAG QA App", layout="wide")
st.title(" Local RAG App with PII Masking + spaCy + FAISS")

uploaded_file = st.file_uploader(" Upload a text file", type=["txt"])
apply_pii_mask = st.checkbox(" Mask PII (Names, Emails, Phone Numbers)")

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")

    if apply_pii_mask:
        raw_text = mask_pii(raw_text)

    chunks = semantic_chunk(raw_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Embed and store in FAISS
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    user_query = st.text_input(" Ask a question about the uploaded content")

    if user_query:
        answer = qa.run(user_query)
        st.markdown("Answer:")
        st.success(answer)

        with st.expander("View retrieved chunks"):
            for doc in vectorstore.similarity_search(user_query, k=3):
                st.code(doc.page_content)

else:
    st.info("Upload a text file to begin.")

