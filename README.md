# Arabic-Official-Documents-OCR
Just a personal space to track my progress on document data extraction. 
 
This is a two-phase Python pipeline that processes images of Arabic documents (Passports, National IDs, etc.). It uses OpenRouter to hit Qwen vision and instruct models to:
1. Classify the document type and extract raw text verbatim.
2. Normalize the Arabic-Indic numbers and map the extracted text into a structured JSON schema.


