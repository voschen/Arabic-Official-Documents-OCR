import requests
# import PIL
import base64
import json
import os
import shutil
from dotenv import load_dotenv
from schemas import DOCUMENT_SCHEMAS

load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")

print(f"API Key Loaded: {API_KEY is not None}")
print(f"API Key Value: {API_KEY[:20]}..." if API_KEY else "API Key is None!")

CONFIDENCE_THRESHOLD = 0.7

# first makes a list of all the files in the input folder
input_folder = "Documents/inputs"
all_entries = os.listdir(input_folder)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
image_files = [f for f in all_entries if f.lower().endswith(image_extensions)] # only takign in images 

full_paths = [os.path.join(input_folder, f) for f in image_files] #converts to full paths


for image_path in full_paths:

    # conversion to base 64
    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8') 

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/jpeg')

    data_url = f"data:{mime};base64,{base64_str}"


    DOCUMENT_TYPES = [
    "Passport",
    "National ID", 
    "Company Registration",
    "Establishment Registration",
    "Driver's License",
    "Residence Permit",
    "Hand Writing"
    ]
    

    phase1_package = {
        "model": "qwen/qwen3-vl-32b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an expert at document vision, analysis, and transcribing. Analyze this document image with high precision.
                        TASK 1: Classify the document.
                        Choose ONLY one type from this list: {DOCUMENT_TYPES}
                        (Include "Hand Writing" if the document is primarily handwritten notes).

                        TASK 2: Extract ALL visible text exactly as it appears.
                        RULES FOR ACCURACY:
                        - Do NOT convert numbers. Keep Arabic-Indic (٠-٩) AND Western (0-9) exactly as seen.
                        - Do NOT translate. Keep Arabic text in Arabic, English in English.
                        - For Handwriting: Pay extra attention to ambiguous characters (e.g., distinguish 1 vs l vs I).
                        - Preserve line breaks and spacing where possible.
                        - If text is unclear or cut off, mark it as "[unclear]".

                        OUTPUT JSON FORMAT:
                        {{
                            "document_type": "string (from the list above)",
                            "confidence": float (0-1),
                            "extracted_text": "string (verbatim extraction of all content)"
                        }}
                                 """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"},

    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "ID OCR",
    }

    url = "https://openrouter.ai/api/v1/chat/completions"

    response = requests.post(url, headers=headers, json=phase1_package)


    if response.status_code == 200:
        # Acquire the response text an convert it to json
        response_text = response.json()['choices'][0]['message']['content']
        data = json.loads(response_text)

        doc_type = data["document_type"]
        confidence = data.get("confidence", 0)
        extracted_text = data["extracted_text"]

        print(f"✅ Phase 1 Complete! Type: {doc_type}, Confidence: {confidence}")

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⚠️ Low confidence ({confidence}), moving to failed/")
            shutil.move(image_path, os.path.join("Documents/failed", os.path.basename(image_path)))
            continue  

        schema = DOCUMENT_SCHEMAS.get(doc_type)

        if not schema:
            print(f"⚠️ Unknown document type: {doc_type}")
            continue
        

        phase2_package = {
            "model": "qwen/qwen-2.5-72b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data extraction expert. Fill the JSON schema using ONLY the provided text."
                },
                {
                    "role": "user",
                    "content": f"""
                    Document Type: {doc_type}

                    Extracted Text (Raw OCR):
                    {extracted_text}

                    INSTRUCTIONS:
                    1. Normalize Numbers: Convert ANY Arabic-Indic digits (٠-٩) to Western digits (0-9).
                    2. Structure Data: Fill the schema fields using the normalized text.
                    3. Validation: If a field value looks incorrect based on context, mark it as null.

                    Schema:
                    {json.dumps(schema, indent=2)}

                    Output ONLY valid JSON matching this schema.
                    """
                }
            ],
            "response_format": {"type": "json_object"}          
        }

        phase2_response = requests.post(url, headers=headers, json=phase2_package)

        if phase2_response.status_code == 200:

            final_text = phase2_response.json()['choices'][0]['message']['content']
            
            # 🛑 CLEAN MARKDOWN WRAPPERS
            final_text = final_text.replace('```json', '').replace('```', '').strip()
            
            # Check if empty after cleaning
            if not final_text:
                print(f"❌ Phase 2 returned empty response!")
                shutil.move(image_path, os.path.join("Documents/failed", os.path.basename(image_path)))
                continue

            structured_data = json.loads(final_text)

            # wrapping structured data with meta data
            entry = {
            "source_image": os.path.basename(image_path),  # Gets filename like "passport.jpg"
            "document_type": doc_type,
            "fields": structured_data  # This is your Phase 2 JSON
            }


            # append to the all data json file
            if os.path.exists("Documents/data.json"):
                with open("Documents/data.json", "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        all_data = json.loads(content)
                    else:
                        all_data = []
            else:
                all_data = []

 
            all_data.append(entry)


            with open("Documents/data.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Phase 2 Complete!")
            print(json.dumps(structured_data, indent=2, ensure_ascii=False))


        else:
            print(f"❌ Phase 2 Failed: {phase2_response.text}") 

    else:
        print("❌ API Failed - Check the error above")

    
