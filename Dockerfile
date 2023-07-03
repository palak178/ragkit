# Specify the base image for the container
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Set environment variables (replace the values with the actual keys)
ENV OPENAI_KEY="YOUR_OPENAI_API_KEY"
ENV QDRANT_KEY="YOUR_QDRANT_API_KEY"
ENV QDRANT_URL="YOUR_QDRANT_URL"

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "streamlit_app.py"]
