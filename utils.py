import io
import base64

def preprocess_document(document_content):
    # Preprocess the document content as needed
    # Convert the document content to text, clean it, etc.
    document_text = ""

    return document_text

def convert_embedding_to_base64(embedding):
    # Convert the embedding to a base64 string
    embedding_bytes = embedding.tobytes()
    embedding_base64 = base64.b64encode(embedding_bytes).decode()

    return embedding_base64

def save_embeddings_to_supabase(document_name, document_embeddings, supabase_client, supabase_table):
    # Create a list of dictionaries representing the document embeddings
    embeddings_data = [
        {"document_name": document_name, "embedding": convert_embedding_to_base64(embedding)}
        for embedding in document_embeddings
    ]

    # Insert the embeddings data into the Supabase table
    response = supabase_client.table(supabase_table).insert(embeddings_data).execute()

    # Check the response status to ensure the data was successfully saved
    if response.status_code == 201:
        print("Embeddings saved to Supabase successfully")
    else:
        print("Failed to save embeddings to Supabase")

def process_document(document_content):
    # Convert the document content to text or process it as required
    document_text = preprocess_document(document_content)

    # Create document embeddings using your chosen method
    document_embeddings = generate_document_embeddings(document_text)

    return document_embeddings

def generate_document_embeddings(document_text):
    # Implement your chosen method to generate document embeddings
    # This can be TF-IDF, word averaging, doc2vec, or any other approach

    # Placeholder code for illustration (replace with your actual implementation)
    embedding = [0.1, 0.2, 0.3]  # Replace with the actual document embedding

    return [embedding]  # Return a list of embeddings (each document can have multiple embeddings)
