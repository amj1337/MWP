from deep_translator import GoogleTranslator
import json

# Load the JSON file
file_path = './dataset/asdiv-a/validset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the fields to translate
fields_to_translate = ["Body", "Question", "Solution-Type", "Answer"]

# Function to recursively translate text fields
def translate_text_fields(entry, fields):
    for field in fields:
        if field in entry:
            if isinstance(entry[field], str):
                try:
                    entry[field] = GoogleTranslator(source='en', target='tr').translate(entry[field])
                except Exception as e:
                    print(f"Error translating field '{field}': {e}")
            elif isinstance(entry[field], dict):
                translate_text_fields(entry[field], entry[field].keys())

# Translate each relevant field in each entry
for entry in data:
    translate_text_fields(entry, fields_to_translate)

# Save the translated data back to a JSON file
translated_file_path = './dataset/asdiv-a/translated_validset.json'
with open(translated_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Translation complete. Translated file saved to:", translated_file_path)
