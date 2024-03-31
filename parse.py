import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict


# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))

log_file_path = 'dataset-v1.jsonl'
dataset_ikm3_path = 'dataset-IKM3.jsonl'
train_ikm2_path = 'trainIKM2.jsonl'
extra_dataset_path = 'extra_dataset.jsonl'  
output_file_path = 'combined_dataset.jsonl'

def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def standardize_entry(entry):
    return {
        "Prompt": entry.get("Prompt", entry.get("text", "")),
        "Response": entry.get("Response", ""),
        "Metadata": entry.get("Metadata", {}),
        "Context and Methodology": entry.get("Context and Methodology", ""),
        "Documentation and Expansion": entry.get("Documentation and Expansion", "")
    }

def extract_keywords(text):
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if not w.lower() in stop_words]
    return list(set(filtered_words))  

def enhance_linking_and_tagging(entries):
    all_keywords = defaultdict(list)
    for i, entry in enumerate(entries):
        text = entry["Prompt"] + " " + entry.get("Response", "")
        keywords = extract_keywords(text)
        entry["Tags"] = keywords
        for keyword in keywords:
            all_keywords[keyword].append(i)
    
    for entry in entries:
        shared_keywords = entry["Tags"]
        related_entries = set()
        for keyword in shared_keywords:
            related_entries.update(all_keywords[keyword])
        
        related_entries.discard(entries.index(entry))
        entry["Related Entries"] = list(related_entries)
    return entries

def merge_and_process_datasets(log_entries, dataset_ikm3, train_ikm2, extra_dataset):
    standardized_entries = [standardize_entry(entry) for entry in log_entries + dataset_ikm3 + train_ikm2 + extra_dataset]
    enhanced_entries = enhance_linking_and_tagging(standardized_entries)
    return enhanced_entries

def write_to_jsonl(entries, output_file_path):
    with open(output_file_path, 'w') as file:
        for entry in entries:
            file.write(json.dumps(entry) + '\n')

def main():
    log_entries = read_jsonl_file(log_file_path)
    dataset_ikm3_entries = read_jsonl_file(dataset_ikm3_path)
    train_ikm2_entries = read_jsonl_file(train_ikm2_path)
    extra_dataset_entries = read_jsonl_file(extra_dataset_path)  

    combined_entries = merge_and_process_datasets(log_entries, dataset_ikm3_entries, train_ikm2_entries, extra_dataset_entries)
    write_to_jsonl(combined_entries, output_file_path)

if __name__ == "__main__":
    main()
