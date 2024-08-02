from deep_translator import GoogleTranslator
import json
import re

# Load the JSON file
file_path = './dataset/mawps_asdiv-a_svamp/testset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the fields to translate
fields_to_translate = ["Question", "Body", "Ques", "question"]

# Function to replace placeholders with unique markers
def replace_placeholders(text):
    placeholders = re.findall(r'number\d+', text)
    unique_markers = {placeholder: f'UNIQUE_PLACEHOLDER_{i}' for i, placeholder in enumerate(placeholders)}
    for placeholder, marker in unique_markers.items():
        text = text.replace(placeholder, marker)
    return text, unique_markers

# Function to restore placeholders from unique markers
def restore_placeholders(text, unique_markers):
    for placeholder, marker in unique_markers.items():
        text = text.replace(marker, placeholder)
    return text

# Function to recursively translate text fields
def translate_text_fields(entry, fields):
    for field in fields:
        if field in entry:
            if isinstance(entry[field], str):
                try:
                    original_text = entry[field]
                    text, unique_markers = replace_placeholders(original_text)
                    translated_text = GoogleTranslator(source='en', target='tr').translate(text)
                    entry[field] = restore_placeholders(translated_text, unique_markers)
                except Exception as e:
                    print(f"Error translating field '{field}': {e}")
            elif isinstance(entry[field], dict):
                translate_text_fields(entry[field], entry[field].keys())

# Translate each relevant field in each entry
for entry in data:
    translate_text_fields(entry, fields_to_translate)

# Save the translated data back to a JSON file
translated_file_path = './dataset/mawps_asdiv-a_svamp/translated_testset.json'
with open(translated_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Translation complete. Translated file saved to:", translated_file_path)
