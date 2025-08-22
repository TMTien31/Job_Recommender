import fitz # PyMuPDF
import os 
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
  

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    
    Args:
        uploaded_file (str): The path to the PDF file.
        
    Returns:
        str: The extracted text.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def ask_llm(prompt, max_tokens=500):
    """
    Asks a language model a question and returns the response.
    
    Args:
        prompt (str): The question to ask the model.
        model (str): The model to use for the query.
        temperature (float): The temperature for the model's response.
        
    Returns:
        str: The model's response.
    """
    response = client.chats.create(
        model= "gemini-2.0-flash-lite",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=max_tokens
    )

    return response.record_history