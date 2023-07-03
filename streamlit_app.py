import streamlit as st
import requests
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
import openai
from qdrant_client.http.exceptions import ResponseHandlingException

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
QDRANT_KEY = os.getenv("QDRANT_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = "embeddings"

# Initialize Langchain client
llm = OpenAI(openai_api_key=OPENAI_KEY)


# Initialize SentenceTransformer model
sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_KEY
)

# Streamlit app
st.title("LLM Chatbot")
st.write("Upload a PDF file")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Process uploaded file
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    content = uploaded_file.read().decode("ISO-8859-1")

    if content:
        # Preprocess and convert to document embeddings
        embeddings = sentence_transformer_model.encode([content])[0]
        st.write("Embeddings generated successfully!")

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
                st.write("File processed and embeddings saved to Qdrant!")
            else:
                st.write("Failed to save embeddings to Qdrant.")
        except ResponseHandlingException as e:
            st.write("Error occurred during the Qdrant upsert operation:")
            st.write(str(e))

# Get Answer tab
st.sidebar.title("Get Answer")
user_question = st.sidebar.text_input("Ask a question")
if user_question:
    try:
        # Encode question
        question_embedding = sentence_transformer_model.encode([user_question])[0]

        # Query Qdrant for relevant documents
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=question_embedding.tolist(),
            limit=5
        )

        # Generate answer using GPT-3.5-turbo model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_question}
            ]
        )

        # Extract the generated answer
        answer = response.choices[0].message.content

        # Display the answer
        st.subheader("Answer:")
        st.write(answer)

    except Exception as e:
        st.write("An error occurred while generating the answer:")
        st.write(str(e))
