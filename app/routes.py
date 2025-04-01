import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .llm import LLMManager
from .asr import ASRManager
from .pdf_processor import PDFProcessor
import tempfile

bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize managers
llm_manager = None
asr_manager = ASRManager()
pdf_processor = PDFProcessor()

def get_llm_manager():
    """Get or initialize the LLM Manager."""
    global llm_manager
    if llm_manager is None:
        # Get DB path from environment or use default
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        llm_manager = LLMManager(db_path)
    return llm_manager

@bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        manager = get_llm_manager()
        response = manager.process_query(message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_path = temp_file.name
            audio_file.save(audio_path)
        
        try:
            # Transcribe the audio using the ASR manager
            transcript = asr_manager.transcribe(audio_path)
            
            # Clean up the temporary file
            os.unlink(audio_path)
            
            return jsonify({'transcript': transcript})
        except Exception as e:
            # Clean up the temporary file in case of error
            os.unlink(audio_path)
            raise e
            
    except Exception as e:
        print(f"Error in transcribe endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/settings', methods=['POST'])
def update_settings():
    """Update LLM settings."""
    try:
        data = request.json
        llm_model = data.get('llmModel')
        asr_model = data.get('asrModel')
        temperature = data.get('temperature')
        
        update_result = {}
        
        if llm_model or temperature is not None:
            manager = get_llm_manager()
            llm_result = manager.update_settings(model=llm_model, temperature=temperature)
            update_result.update(llm_result or {})
            
        if asr_model:
            # Just update the frontend model setting
            # The backend always uses NIVED47/Conformer_sanskrit
            asr_result = asr_manager.update_settings(model=asr_model)
            update_result.update(asr_result or {})
            
        return jsonify({
            'status': 'Settings updated successfully',
            'settings': update_result
        })
    except Exception as e:
        print(f"Error in settings endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file uploads."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        pdf_file = request.files['file']
        
        # Check if file is a PDF
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Uploaded file must be a PDF'}), 400
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            pdf_path = temp_file.name
            pdf_file.save(pdf_path)
        
        try:
            # Process the PDF and add to context
            text = pdf_processor.extract_text(pdf_path)
            manager = get_llm_manager()
            manager.add_document_to_context(text)
            
            # Clean up the temporary file
            os.unlink(pdf_path)
            
            return jsonify({'message': 'PDF uploaded and processed successfully'})
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
            raise e
            
    except Exception as e:
        print(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear chat history."""
    try:
        manager = get_llm_manager()
        manager.clear_history()
        return jsonify({'status': 'Chat history cleared successfully'})
    except Exception as e:
        print(f"Error in clear-history endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500 