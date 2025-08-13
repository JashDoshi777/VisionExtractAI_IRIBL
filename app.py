# app.py

import os
import io
import json
import hashlib
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import fitz  # PyMuPDF
import rag_core 

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'user_data' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- Helper Functions for User-Specific Data ---

def _get_user_data_filepath(user_api_key, mode):
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    return os.path.join(DATA_FOLDER, f'{user_hash}_{mode}_data.json')

def _load_user_data(user_api_key, mode):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []
    return []

def _save_user_data(user_api_key, mode, data):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except IOError as e:
        print(f"❌ Could not save data to {filepath}! Error: {e}")
        return False

# --- Business Card Logic (UNCHANGED) ---

def extract_card_data(image_bytes, user_api_key):
    print("🤖 Processing business card with Gemini Vision API...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = """
        You are an expert at reading business cards. Analyze the image and extract information into a structured JSON format.
        The JSON object must use these exact keys: "Owner Name", "Company Name", "Email", "Number", "Address".
        If a piece of information is not present, its value must be `null`.
        Your entire response MUST be a single, valid JSON object.
        """
        response = model.generate_content([prompt, img])
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        parsed_info = json.loads(json_text)
        return {
            "Owner Name": parsed_info.get("Owner Name"),
            "Company Name": parsed_info.get("Company Name"),
            "Email": parsed_info.get("Email"),
            "Number": parsed_info.get("Number"),
            "Address": parsed_info.get("Address"),
        }
    except Exception as e:
        print(f"❌ Error during Gemini API call for business card: {e}")
        return {"error": f"Failed to parse AI response: {e}"}

# --- Brochure Logic ---

def extract_brochure_contacts_and_company(image_list, user_api_key):
    """Step 1 of the chain: Meticulously extract only the contacts and company name."""
    print("🤖 Brochure Step 1: Extracting contacts and company...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        prompt = """
        You are a world-class data extraction expert. Your ONLY task is to find two pieces of information from the provided brochure images:
        1. The primary **Company Name**.
        2. A list of **every individual person** mentioned. For each person, you MUST extract their "Owner Name", "Email", and "Number". It is CRITICAL that you find the phone number if it exists.

        Return ONLY a single JSON object with two keys: "company_name" and "contacts". The "contacts" key must be a LIST of JSON objects.
        If any field for a person is not found, use `null`. If no people are found, return an empty list for "contacts".
        Do NOT include any other information or text.
        """
        
        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ Error during brochure contact extraction: {e}")
        return {"error": f"Failed to parse contacts from brochure: {e}"}

def extract_brochure_raw_text(image_list, contacts_to_exclude, user_api_key):
    """Step 2 of the chain: Extract all other text, EXCLUDING the contacts we already found."""
    print("🤖 Brochure Step 2: Extracting remaining raw text...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        exclude_string = json.dumps(contacts_to_exclude, indent=2)
        prompt = f"""
        You are a text summarization expert. Analyze the provided brochure images.
        You have been given a list of specific contact details that have already been extracted. This list is:
        ---
        {exclude_string}
        ---
        Your task is to extract all the OTHER text from the brochure. This includes marketing copy, slogans, mission statements, general addresses, product descriptions, etc.
        Your response MUST NOT include any of the specific names, emails, or phone numbers from the list provided above.
        Return a single JSON object with one key: "raw_text".
        """
        
        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ Error during brochure raw text extraction: {e}")
        return {"error": f"Failed to parse raw text from brochure: {e}"}

# --- Processing Endpoints ---

@app.route('/process_card', methods=['POST'])
def process_card_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        extracted_info = extract_card_data(image_bytes, user_api_key)
        if "error" in extracted_info: return jsonify(extracted_info), 500

        file_id = os.urandom(8).hex()
        _, f_ext = os.path.splitext(file.filename)
        safe_ext = f_ext if f_ext.lower() in ['.png', '.jpg', '.jpeg', '.webp'] else '.png'
        image_filename = f"{file_id}{safe_ext}"
        save_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(save_path, 'wb') as f: f.write(image_bytes)
        
        extracted_info['id'] = file_id
        extracted_info['image_filename'] = image_filename
        
        user_contacts = _load_user_data(user_api_key, 'cards')
        user_contacts.insert(0, extracted_info)
        _save_user_data(user_api_key, 'cards', user_contacts)
        
        raw_text_for_rag = ' '.join(str(v) for k, v in extracted_info.items() if v and k not in ['id', 'image_filename'])
        rag_core.add_document_to_knowledge_base(user_api_key, raw_text_for_rag, file_id, 'cards')
        
        return jsonify(extracted_info)
    except Exception as e:
        print(f"An error occurred in process_card endpoint: {e}")
        return jsonify({'error': 'Server processing failed'}), 500

@app.route('/process_brochure', methods=['POST'])
def process_brochure_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400

    try:
        pdf_bytes = file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        brochure_id = os.urandom(8).hex()
        pdf_filename = f"{brochure_id}.pdf"
        save_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        with open(save_path, 'wb') as f: f.write(pdf_bytes)

        image_list = [Image.open(io.BytesIO(page.get_pixmap().tobytes("png"))) for page in pdf_doc]
        
        if not image_list:
            return jsonify({'error': 'No images found in the PDF brochure.'}), 400
            
        # FIXED: Corrected the function name here
        contact_data = extract_brochure_contacts_and_company(image_list, user_api_key)
        if "error" in contact_data: return jsonify(contact_data), 500
        
        raw_text_data = extract_brochure_raw_text(image_list, contact_data.get("contacts", []), user_api_key)
        if "error" in raw_text_data: return jsonify(raw_text_data), 500

        final_brochure_object = {
            "id": brochure_id,
            "company_name": contact_data.get("company_name", "Unknown Company"),
            "contacts": contact_data.get("contacts", []),
            "raw_text": raw_text_data.get("raw_text", ""),
            "image_filename": pdf_filename
        }

        for contact in final_brochure_object["contacts"]:
            contact["id"] = os.urandom(8).hex()
        
        user_brochures = _load_user_data(user_api_key, 'brochures')
        user_brochures.insert(0, final_brochure_object)
        _save_user_data(user_api_key, 'brochures', user_brochures)
        
        raw_text_for_rag = final_brochure_object.get("raw_text", "")
        contacts_for_rag = final_brochure_object.get("contacts", [])
        if contacts_for_rag:
            raw_text_for_rag += "\n\n--- Extracted Contacts ---\n"
            for contact in contacts_for_rag:
                name = contact.get("Owner Name", "N/A")
                email = contact.get("Email", "N/A")
                number = contact.get("Number", "N/A")
                raw_text_for_rag += f"Name: {name}, Email: {email}, Number: {number}\n"
        
        rag_core.add_document_to_knowledge_base(user_api_key, raw_text_for_rag, brochure_id, 'brochures')
        
        return jsonify(final_brochure_object)

    except Exception as e:
        print(f"An error occurred in process_brochure endpoint: {e}")
        return jsonify({'error': f'Server processing failed: {e}'}), 500

# --- Shared Endpoints ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    user_api_key = data.get('apiKey')
    query_text = data.get('query')
    mode = data.get('mode') 

    if not all([user_api_key, query_text, mode]):
        return jsonify({'error': 'API key, query, and mode are required.'}), 400

    try:
        data_source = _load_user_data(user_api_key, mode)
        
        if "table" in query_text.lower() or "list all" in query_text.lower():
            intent = 'synthesis'
        else:
            intent = 'research'
        
        print(f"🤖 Intent detected: {intent}")

        if intent == 'synthesis':
            synthesis_data = []
            if mode == 'brochures':
                for brochure in data_source:
                    for contact in brochure.get('contacts', []):
                        synthesis_data.append({
                            "Company Name": brochure.get("company_name"),
                            "Owner Name": contact.get("Owner Name"),
                            "Email": contact.get("Email"),
                            "Number": contact.get("Number")
                        })
            else:
                synthesis_data = data_source

            synthesis_prompt = f"""
            You are a data analysis assistant. Based on the JSON data below, fulfill the user's request. Format tables as clean Markdown.
            JSON Data: {json.dumps(synthesis_data, indent=2)}
            User Request: {query_text}
            Answer:
            """
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(synthesis_prompt)
            answer = response.text
        else:
            answer = rag_core.query_knowledge_base(user_api_key, query_text, mode)

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"❌ Error in /chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/load_data/<mode>', methods=['POST'])
def load_data_endpoint(mode):
    user_api_key = request.json.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if mode not in ['cards', 'brochures']: return jsonify({'error': 'Invalid mode specified'}), 400
    user_data = _load_user_data(user_api_key, mode)
    return jsonify(user_data)

@app.route('/update_card/<mode>/<item_id>', methods=['POST'])
def update_card_endpoint(mode, item_id):
    data = request.get_json()
    user_api_key = data.get('apiKey')
    field, value = data.get('field'), data.get('value')
    contact_id = data.get('contactId')

    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    
    user_data = _load_user_data(user_api_key, mode)
    
    if mode == 'cards':
        for contact in user_data:
            if contact.get('id') == item_id:
                contact[field] = value
                _save_user_data(user_api_key, mode, user_data)
                rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                new_text = ' '.join(str(v) for k, v in contact.items() if v and k not in ['id', 'image_filename'])
                rag_core.add_document_to_knowledge_base(user_api_key, new_text, item_id, mode)
                return jsonify({"success": True})
    
    elif mode == 'brochures':
        for brochure in user_data:
            if brochure.get('id') == item_id:
                for contact in brochure.get('contacts', []):
                    if contact.get('id') == contact_id:
                        contact[field] = value
                        _save_user_data(user_api_key, mode, user_data)
                        rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                        rag_core.add_document_to_knowledge_base(user_api_key, brochure['raw_text'], item_id, mode)
                        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route('/delete_card/<mode>/<item_id>', methods=['DELETE'])
def delete_card_endpoint(mode, item_id):
    user_api_key = request.json.get('apiKey')
    contact_id = request.json.get('contactId')

    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400

    user_data = _load_user_data(user_api_key, mode)
    
    if mode == 'cards':
        original_len = len(user_data)
        user_data = [c for c in user_data if c.get('id') != item_id]
        if len(user_data) < original_len:
            _save_user_data(user_api_key, mode, user_data)
            rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
            return jsonify({"success": True})

    elif mode == 'brochures':
        if contact_id:
            for brochure in user_data:
                original_len = len(brochure.get('contacts', []))
                brochure['contacts'] = [c for c in brochure.get('contacts', []) if c.get('id') != contact_id]
                if len(brochure.get('contacts', [])) < original_len:
                    _save_user_data(user_api_key, mode, user_data)
                    return jsonify({"success": True})
        else:
            original_len = len(user_data)
            user_data = [b for b in user_data if b.get('id') != item_id]
            if len(user_data) < original_len:
                _save_user_data(user_api_key, mode, user_data)
                rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                return jsonify({"success": True})

    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    rag_core.initialize_rag_system()
    print("--- Server is starting! ---")
    print(f"User-specific data will be saved in '{os.path.abspath(DATA_FOLDER)}'")
    print("To use the dashboard, open your web browser and go to: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
