import os
from flask import (Flask, request, render_template, send_file,
                   redirect, url_for, flash)

# --- LLM, TTS, and File Reader Imports ---
from transformers import pipeline
import pyttsx3
import PyPDF2
import docx

# --- 1. APP CONFIGURATION ---

app = Flask(__name__)

# You MUST change this secret key for security
app.config['SECRET_KEY'] = 'a_very_secret_random_key_f0r_pr0ducti0n'

# Folder setup
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. MODEL LOADING (SPEED OPTIMIZED) ---

print("Loading summarization model... This may take a moment.")
# We use 'distilbart' which is MUCH faster than the 'bart-large' model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
print("Model loaded successfully.")


# --- 3. HELPER FUNCTIONS (Methodology) ---

def read_text_file(filepath):
    """
    Reads content from .txt, .pdf, and .docx files.
    """
    text = ""
    if filepath.endswith(".txt"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()

    elif filepath.endswith(".pdf"):
        try:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return "Could not read PDF file."

    elif filepath.endswith(".docx"):
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return "Could not read DOCX file."

    else:
        return "Unsupported file type. Please upload .txt, .pdf, or .docx"
    return text


def summarize_text(text):
    """
    Handles long text by batch chunking.
    """
    if not text or not text.strip():
        return "Input text was empty. No summary generated."

    # Safer 3000 character chunk size (prevents IndexError)
    max_chunk_size = 3000

    if len(text) <= max_chunk_size:
        # If text is short enough, summarize as normal
        print("Text is short. Summarizing directly...")
        summary_list = summarizer(text,
                                  max_length=150,
                                  min_length=30,
                                  do_sample=False,
                                  truncation=True)
        return summary_list[0]['summary_text']

    else:
        # If text is too long, split into chunks
        print(f"Text is long. Splitting into chunks...")
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        print(f"Summarizing {len(chunks)} chunks using batch processing...")

        # Pass the entire list of chunks at once for batch processing (faster)
        chunk_summary_list_of_dicts = summarizer(chunks,
                                                 max_length=300,
                                                 min_length=100,
                                                 do_sample=False,
                                                 truncation=True)

        chunk_summaries = [summary_dict['summary_text'] for summary_dict in chunk_summary_list_of_dicts]

        # Join all the chunk summaries into one big text
        final_summary = "\n".join(chunk_summaries)
        print("Joined summaries.")
        return final_summary


# --- 4. CORE APP ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file upload and displays results."""

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            # --- 1. Read File ---
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            original_text = read_text_file(filepath)

            # --- 2. Summarize ---
            summary_text = summarize_text(original_text)

            # --- 3. Convert to Audio (pyttsx3) ---
            # Create a simple, unique filename
            audio_filename = f"summary_{file.filename.split('.')[0]}.mp3"
            audio_path_full = os.path.join(OUTPUT_FOLDER, audio_filename)

            try:
                engine = pyttsx3.init()
                engine.save_to_file(summary_text, audio_path_full)
                engine.runAndWait()
            except Exception as e:
                print(f"Audio generation failed: {e}")
                flash(f"Audio generation failed: {e}", "danger")

            # --- 4. Show Results ---
            # Render the result template with the summary and audio file
            return render_template('dashboard.html',
                                   summary=summary_text,
                                   audio_file=audio_filename)

    # This is the 'GET' request (when the user first visits)
    return render_template('dashboard.html')



# --- 5. File Serving ---
@app.route('/static/<filename>')
def serve_audio(filename):
    """Lets the HTML <audio> tag find the generated MP3 file."""
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


# --- 6. Run the app ---
if __name__ == "__main__":
    app.run(debug=True)