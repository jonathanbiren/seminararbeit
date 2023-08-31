import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os


def get_pdf_texts(pdf_doc):
    text = ''
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len
    )

    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    # * Here we pass the user question to the conversation chain that we created
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # * Initializing session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs')
    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        # * We upload our PDF document
        pdf_doc = st.file_uploader(
            'Upload your PDFs here',
            accept_multiple_files=False)
        if st.button('Process'):
            with st.spinner('Processing'):
                vectorstore = None
                documentName = pdf_doc.name[:-4]
                # * Check whether the vectorstore for the document already exists on our drive
                if os.path.exists(f'./Embeddings/{documentName}.pkl'):
                    with open(f'./Embeddings/{documentName}.pkl', 'rb') as f:
                        vectorstore = pickle.load(f)
                        st.write('Loaded Vectorstore from drive.')

                # * In case we don't have vectorstore already saved, we create it from the PDF
                else:
                    # * get PDF
                    raw_text = get_pdf_texts(pdf_doc)
                    # * get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # * create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    with open(f"./Embeddings/{documentName}.pkl", "wb") as f:
                        # * Locally safe vectorstore on disk
                        pickle.dump(vectorstore, f)
                        st.write('Created and saved vectorstore')

                # * create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
