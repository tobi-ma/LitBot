# LitBot - A Streamlit Literature Research Engine

LitBot is a powerful tool designed to assist users in querying and retrieving context-based answers from a collection of PDF documents. This project leverages advanced natural language processing (NLP) models and vector databases to provide accurate and relevant responses to user questions.

## Features

- **Document Retrieval**: Load and process multiple PDF documents for querying.
- **Conversational Interface**: Interact with the engine through a user-friendly chat interface.
- **Configurable Response Length**: Choose between short, medium-short, medium-long, and long responses.
- **Temperature Adjustment**: Control the creativity of the responses by adjusting the model's temperature.
- **Save and Load Chat History**: Save your conversation history and reload it for future sessions.
- **Index Rebuilding**: Rebuild the vector index to accommodate new or changed documents.

## Requirements

The project requires the following Python packages:

- **streamlit**
- **langchain-huggingface**
- **langchain-community**
- **fitz**
- **PyMuPDF**
- **transformers**
- **torch**
- **sentence-transformers**

## Installation

1. Clone the repository:


    git clone https://github.com/tobi-ma/litbot.git
    cd literature-research-engine


2. Install the required packages:

    
    pip install -r requirements.txt


3. Set up your Hugging Face API token:

    Ensure you have your Hugging Face API token available and set it as an environment variable:

    
    export HUGGINGFACEHUB_API_TOKEN='your_hugging_face_api_token'


## Usage

1. Prepare your PDF documents:

    Place your PDF documents in the literature_data directory on root. The engine will process all PDFs in this directory.


2. Run the Streamlit application:

    
    streamlit run app.py


3. Interact with the interface:

    - Use the sidebar to adjust the number of documents to retrieve, the model's temperature, and the response length.
    - Enter your questions in the chat input box at the bottom of the page.
    - Click "Read Aloud" to have the assistant's response read aloud.
    - Save and load chat history using the respective buttons in the sidebar.
    - Rebuild the index if you add or change documents in the literature_data directory.

# Functionality
## Loading Documents

The engine loads all PDF documents placed in the literature_data directory. It processes each document and stores the text content for querying.

## Rebuilding the Index

If the documents in the literature_data directory change, you may need to rebuild the vector index. Use the "Rebuild Index" button in the sidebar to start this process. Be aware that rebuilding the index can be time-consuming.

## Chat Interface

The chat interface allows you to ask questions and receive context-based answers from the loaded documents. Adjust the response length and temperature to tailor the responses to your needs.

## Saving and Loading Chat History

You can save your chat history to a file and load it in future sessions. This feature ensures continuity and allows you to refer back to previous interactions.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## Acknowledgements

- Hugging Face for providing powerful NLP models.
- Streamlit for the interactive web application framework.
- PyMuPDF for PDF document processing.