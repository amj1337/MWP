from deep_translator import GoogleTranslator
import json

# Load the JSON file
file_path = './dataset/mawps_asdiv-a_svamp/trainset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the fields to translate
fields_to_translate = ["Question", "Body", "Ques", "question"]

# Translate each relevant field in each entry
for entry in data:
    for field in fields_to_translate:
        if field in entry and entry[field]:  # Check if field exists and is not None
            try:
                translated_text = GoogleTranslator(source='en', target='tr').translate(entry[field])
                entry[field] = translated_text
            except Exception as e:
                print(f"Error translating field '{field}' in entry {entry.get('id', 'unknown')}: {e}")

# Save the translated data back to a JSON file
translated_file_path = './dataset/mawps_asdiv-a_svamp/translated_trainset.json'
with open(translated_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Translation complete. Translated file saved to:", translated_file_path)
