import streamlit as st
import os
import openai
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
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_KEY
)

# Streamlit app
st.title("LLM Chatbot")

# Sidebar - File upload
st.sidebar.subheader("Upload a PDF file")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Main content
st.subheader("Chat")

# Conversation history
conversation = []

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

# Answer user's questions
user_input = st.text_area("User Input", value='', height=100, key='user_input')

if st.button("Ask"):
    if user_input.strip() != '':
        # Add user question to conversation history
        conversation.append(("user", user_input))

        # Generate answer using OpenAI's GPT-3.5 API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input + "\nAnswer:",
            max_tokens=1500,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        if response.choices and response.choices[0].text.strip():
            answer = response.choices[0].text.strip()

            # Add bot answer to conversation history
            conversation.append(("bot", answer))

# Display conversation history
for role, message in conversation:
    if role == "user":
        conversation_markdown = f"**User:** {message}"
    else:
        conversation_markdown = f"**Bot:** {message}"
    st.markdown(conversation_markdown, unsafe_allow_html=True)

# Add a button to clear the conversation history
if st.button("Clear conversation"):
    conversation.clear()
