# import streamlit as st
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.vectorstores import Chroma
# from streamlit_chat import message
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
import tempfile
import pandas as pd
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")
# Load CSV data
file = r'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

# Set up OpenAI embeddings and create vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore = FAISS.from_texts(texts, embeddings)
# persist_directory = r'dealer1\inventry'
# persist_directory = os.path.join(os.getcwd(), 'dealer1', 'inventry')
# vectordb = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embeddings
# )
retriever = vectorstore.as_retriever(search_type="similarity", k=8)
# qa = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
#     retriever=retriever
# )
# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
#     retriever=vectorstore.as_retriever())
# Streamlit UI setup
st.info(" We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing a environment to check offerings. Our website [engane.ai](https://funnelai.com/).This test application answers about Inventry, Business details, Financing and offers. [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is a inventry dataset.")  

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []

# Initialize conversation history with intro_prompt
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in english language.' If you do not know the answer reply with 'I am sorry'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k'),
    retriever=retriever,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT
#     return_source_documents=True
)
response_container = st.container()
container = st.container()
chat_history=[] 
def conversational_chat(query):
    
    result = qa({"question": query, "chat_history": chat_history})
    st.session_state.history.append((query, result["answer"]))
    return result["answer"]
# conversational_chat(final_prompt)
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input:
        output = conversational_chat(user_input)
        with response_container:
            for i, (query, answer) in enumerate(st.session_state.history):
                message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(answer, key=f"{i}_answer", avatar_style="thumbs")
# with container:
#     with st.form(key='my_form', clear_on_submit=True):
#         user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
#         submit_button = st.form_submit_button(label='Send')
        
#     if submit_button and user_input:
#         output = conversational_chat(user_input)

#         with response_container:
#             for i in range(len(st.session_state.generated)):
#                 message(st.session_state.past[i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                 message(st.session_state.generated[i], key=str(i), avatar_style="thumbs")
