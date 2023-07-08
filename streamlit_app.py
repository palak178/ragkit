import os
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_KEY = os.getenv('OPENAI_KEY')
QDRANT_KEY = os.getenv('QDRANT_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_COLLECTION_NAME = "embeddings"

# Initialize OpenAI API
openai.api_key = OPENAI_KEY

# Initialize SentenceTransformer model
sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

# Initialize the language model and memory objects
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_KEY)
buffer_memory = ConversationBufferWindowMemory(k=3)
conversation = ConversationChain(llm=llm, memory=buffer_memory)

# Streamlit app
st.title("LLM Chatbot")

# Sidebar - File upload
st.sidebar.subheader("Upload a PDF file")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Main content
st.subheader("Chat")

# Reset chat button
if st.button("Reset Chat"):
    st.session_state['conversation'] = []

# Conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Process uploaded file
if uploaded_file is not None:
    content = uploaded_file.read().decode("ISO-8859-1")
    if content:
        # Preprocess and convert to document embeddings
        embeddings = sentence_transformer_model.encode([content])[0]

        # Create a single point for the entire document
        point = PointStruct(id=1, vector=embeddings.tolist())

        try:
            # Insert point into Qdrant collection
            operation_info = qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                wait=True,
                points=[point]
            )

            if operation_info:
                # st.success("File processed and embeddings saved to Qdrant!")
                pass
            else:
                st.error("Failed to save embeddings to Qdrant.")
        except ResponseHandlingException as e:
            st.error("Error occurred during the Qdrant upsert operation:")
            st.error(str(e))

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Generate answer using OpenAI's GPT-3.5 API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt + "\nAnswer:",
        max_tokens=1500,
        temperature=0.5,
        top_p=1.0,
        n=1,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    if response.choices and response.choices[0].text.strip():
        answer = response.choices[0].text.strip()

        # Add user message and bot answer to conversation history
        st.session_state['conversation'].append({"role": "user", "content": prompt})
        st.session_state['conversation'].append({"role": "assistant", "content": answer})

# Display conversation history
for message in st.session_state['conversation']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
