import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    import requests

    url = "https://deep-translate1.p.rapidapi.com/language/translate/v2"

    payload = {
        "q": response,
        "source": "te",
        "target": "en"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "eda5923de4mshd2f68353251ed51p14d85ajsn2bb840bc3f81",
        "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    user_question=response.json()
    docs = new_db.similarity_search(user_question["data"]["translations"]["translatedText"])

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    import requests
    dup=response
    url = "https://deep-translate1.p.rapidapi.com/language/translate/v2"

    payload = {
        "q": response,
        "source": "en",
        "target": "te"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "eda5923de4mshd2f68353251ed51p14d85ajsn2bb840bc3f81",
        "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    resp=response.json()
    print(resp)
    print(resp)
    st.write("Reply: ", resp["data"]["translations"]["translatedText"])
    st.write("English: ", dup)




def main():
    st.set_page_config("Bhagavad-gita chatbot")
    st.header("Chat with Bhagavad-gita")

    user_question = st.text_input("Ask a Question from Bhagavad-gita")

    if user_question:
        user_input(user_question)

    



if __name__ == "__main__":
    main()
