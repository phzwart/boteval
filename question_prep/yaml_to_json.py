import yaml
import json
import sys
from pathlib import Path

def convert_yaml_to_json(yaml_file):
    # Read YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract only id, question and answer
    simplified_data = []
    for qa_pair in data['qa_pairs']:
        simplified_data.append({
            'id': qa_pair['id'],
            'question': qa_pair['question'],
            'answer': qa_pair['answer']
        })
    
    # Write to JSON file
    output_file = Path(yaml_file).stem + '.json'
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    print(f"Converted {yaml_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yaml_to_json.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    convert_yaml_to_json(yaml_file) 