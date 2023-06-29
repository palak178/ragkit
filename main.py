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
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_chat import message
from utils import *
import supabase

# Initialize ChatGPT model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="YOUR_OPENAI_API_KEY")

# Initialize Supabase client
supabase_url = "https://ffcrjxswlivnhtlkjyym.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZmY3JqeHN3bGl2bmh0bGtqeXltIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODgwMzYyNTksImV4cCI6MjAwMzYxMjI1OX0.EL_nf3hQiuwGUcAF4JKkafFGtR75c3eoK4wMuOt_8GE"
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Initialize ConversationChain and buffer memory
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)

# Streamlit app code
st.subheader("Chatbot with Langchain, ChatGPT, and Streamlit")

# File upload functionality
uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

if uploaded_file is not None:
    document_name = uploaded_file.name
    document_content = io.BytesIO(uploaded_file.read())

    # Process the document and extract embeddings
    document_embeddings = process_document(document_content)

    # Save document embeddings to Supabase
    save_embeddings_to_supabase(document_name, document_embeddings)

    st.success("Document uploaded and embeddings saved to Supabase.")

# Helper function to process the document and extract embeddings
def process_document(document_content):
    # Convert the document content to text or process it as required
    document_text = preprocess_document(document_content)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the document text
    vectorizer.fit([document_text])

    # Transform the document text into embeddings
    document_embeddings = vectorizer.transform([document_text]).toarray()

    return document_embeddings

# Helper function to preprocess the document content
def preprocess_document(document_content):
    # Convert the document content to text or preprocess it as required
    document_text = ""

    return document_text

# Helper function to save document embeddings to Supabase
def save_embeddings_to_supabase(document_name, document_embeddings):
    # Create a list of dictionaries representing the document embeddings
    embeddings_data = [
        {"document_name": document_name, "embedding": embedding.tolist()}
        for embedding in document_embeddings
    ]

    # Insert the embeddings data into the Supabase table
    response = supabase_client.table("Documents").insert(embeddings_data).execute()

    # Check the response status to ensure the data was successfully saved
    if response.status_code == 201:
        print("Embeddings saved to Supabase successfully")
    else:
        print("Failed to save embeddings to Supabase")
