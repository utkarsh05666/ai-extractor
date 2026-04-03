import os
import json
import logging
import base64
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pymupdf  
import ollama

logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentIntelligence:
    def __init__(self, model="llama3.2", vision_model="llama3.2-vision"):
        self.model = model
        self.vision_model = vision_model
        self.results = []

    def _validate_extraction(self, data):
        """Bonus: Output Validation & Type Correction"""
        flags = []
        fields = data.get("extracted_fields", {})
        
        
        amount = fields.get("total_amount")
        if isinstance(amount, str):
            try:
                clean_amt = re.sub(r'[^\d.]', '', amount)
                fields["total_amount"] = float(clean_amt)
            except ValueError:
                flags.append(f"Validation Error: total_amount '{amount}' is not numeric.")
        
        
        for date_key in ["date", "due_date"]:
            date_val = fields.get(date_key)
            if date_val and date_val != "null" and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_val)):
                flags.append(f"Validation Error: {date_key} '{date_val}' is not in YYYY-MM-DD.")
            
        return flags

    def _extract_content(self, path):
        """Requirement: Handle Corrupt/Empty Files & OCR Support"""
        try:
            with pymupdf.open(path) as doc:
                text = " ".join([page.get_text() for page in doc]).strip()
                if not text:
                    pix = doc[0].get_pixmap()
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode('utf-8')
                    return None, img_b64
                return text, None
        except Exception as e:
            logging.error(f"File Error {path}: {str(e)}")
            return None, "ERROR_CORRUPT"

    def _call_llm(self, text, filename, image_b64=None, retries=2):
        """Bonus: Strict Schema Enforcement with Reinforced Summary Prompt"""
        active_model = self.vision_model if image_b64 else self.model
        
        
        prompt = f"""
        Analyze the document '{filename}'. 
        
        TASK:
        1. Write a 3-sentence summary: Mention this is an invoice FROM the vendor, the USD total, and the due date.
        2. Extract the structured fields.

        Return ONLY a JSON object with this exact structure:
        {{
            "summary": "Your 3-sentence summary goes here",
            "doc_type": "invoice",
            "extracted_fields": {{
                "invoice_number": "string",
                "vendor_name": "string",
                "date": "YYYY-MM-DD",
                "due_date": "YYYY-MM-DD",
                "total_amount": 0.0,
                "currency": "USD",
                "line_items": [
                    {{"description": "string", "quantity": 0, "price": 0.0}}
                ]
            }}
        }}

        TEXT CONTENT:
        {text if text else "See image for OCR."}
        """

        for attempt in range(retries + 1):
            try:
                images = [image_b64] if image_b64 else None
                response = ollama.generate(model=active_model, prompt=prompt, format="json", images=images)
                raw_data = json.loads(response['response'])
                
                # Check for nested summary
                if not raw_data.get("summary") or raw_data.get("summary") == "N/A":
                    if "summary" in raw_data.get("extracted_fields", {}):
                        raw_data["summary"] = raw_data["extracted_fields"].pop("summary")
                
                return raw_data, []
            except Exception as e:
                if attempt == retries:
                    return None, [f"LLM Error: {str(e)}"]
                time.sleep(1)

    def process_file(self, file_name, folder):
        path = os.path.join(folder, file_name)
        text, scan_data = self._extract_content(path)

        if scan_data == "ERROR_CORRUPT":
            return {"file": file_name, "doc_type": "error", "errors": ["File corrupt"]}
        if not text and not scan_data:
            return {"file": file_name, "doc_type": "skipped", "errors": ["Empty document"]}

        data, errors = self._call_llm(text, file_name, image_b64=scan_data)
        
        if not data:
            return {"file": file_name, "errors": errors}

        val_errors = self._validate_extraction(data)
        
        
        extracted = data.get("extracted_fields", {})
        final_summary = data.get("summary")
        
        if not final_summary or len(final_summary) < 15:
            final_summary = f"Invoice {extracted.get('invoice_number', 'N/A')} from {extracted.get('vendor_name', 'Vendor')} totaling ${extracted.get('total_amount', '0.00')}. Due date is {extracted.get('due_date', 'N/A')}."

        return {
            "file": file_name,
            "doc_type": data.get("doc_type", "unknown"),
            "extracted_fields": extracted,
            "summary": final_summary,
            "confidence": "high" if not (val_errors or errors) else "low",
            "errors": val_errors + errors
        }

    def run(self, input_folder):
        if not os.path.exists(input_folder):
            os.makedirs(input_folder)
            return

        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.pdf', '.txt'))]
        if not files:
            print("No files found in /docs.")
            return

        print(f" Processing {len(files)} files...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.results = list(executor.map(lambda f: self.process_file(f, input_folder), files))

        with open("structured_output.json", "w") as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Success! Results saved to structured_output.json")

if __name__ == "__main__":
    DocumentIntelligence().run("./docs")