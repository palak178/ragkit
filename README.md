# LLM Chatbot

This is a chatbot application powered by OpenAI's GPT-3.5 Turbo language model. The chatbot can answer user questions and provide relevant responses based on the trained model.

## Features

- Chat with the LLM (Language Learning Model) chatbot
- Upload PDF documents for processing
- Extract document embeddings using SentenceTransformer
- Store document embeddings in Qdrant for future retrieval
- Retrieve answers based on the stored embeddings

## Installation

1. Clone the repository:

```
git clone <repository_url>
```
Install the required dependencies:

```
pip install -r requirements.txt
```

Set up the environment variables:

 - OPENAI_KEY: Your OpenAI API key
 - QDRANT_KEY: Your Qdrant API key
 - QDRANT_URL: URL of your Qdrant instance

Run the Streamlit app:
```
streamlit run app.py
```

## Usage

 1. Access the Streamlit app in your web browser.

 2. Upload a PDF file by clicking on the "Upload a PDF file" section in the sidebar.

 3. The uploaded document will be processed, and the embeddings will be stored in Qdrant.

 4. Enter your message in the chat input field and press Enter to send it to the chatbot.

 5. The chatbot will generate a response based on the input using the GPT-3.5 Turbo model.

 6. The conversation history will be displayed in the chat area.

 7. To reset the chat, click the "Reset Chat" button.

## App Link

Access the deployed Streamlit app here

[https://your-app-url.com](https://llmchatbot-dsazs9cvyf.streamlit.app/)https://llmchatbot-dsazs9cvyf.streamlit.app/



