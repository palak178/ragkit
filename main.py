from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from chromadb import ChromaDB
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, ChromaDB, and Streamlit")

# Initialize ChatGPT model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="***REMOVED***")

# Initialize SentenceTransformer model
sentence_transformer_model = SentenceTransformer("LLM chatbot")

# Initialize ChromaDB
chromadb = ChromaDB("Your-ChromaDB-Path")

# Initialize ConversationChain and buffer memory
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
text_container = st.container()

# File upload section
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file is not None:
    file_contents = uploaded_file.read()

    with text_container:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("Typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = find_match(refined_query)

                # Encode the query using SentenceTransformer
                query_embedding = sentence_transformer_model.encode([context])

                # Retrieve most similar document embeddings from ChromaDB
                similar_docs = chromadb.search(query_embedding, top_k=3)

                # Extract the text of similar documents
                similar_texts = [text for _, text in similar_docs]

                # Perform chatbot prediction
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}", documents=similar_texts)

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
