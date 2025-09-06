import json
import os

def extract_questions_by_type(input_file, limit_per_type=100):
    """
    Extracts a specified number of questions for 'pure-text', 'chart/figure', 
    and 'table' types from a .jsonl file.

    The script reads the file line by line and stops once the limit for all 
    categories has been reached to ensure efficiency.

    Args:
        input_file (str): The path to the input annotations.jsonl file.
        limit_per_type (int): The number of questions to collect for each type.
    """
    
    # --- 1. Initialization ---
    # Define the target categories and their corresponding type strings from the JSON
    # Note: We group 'Chart' and 'Figure' together.
    categories = {
        'pure-text': {'types': ["['Pure-text (Plain-text)']"], 'questions': []},
        #'multimodal': {'types': ["['Chart']", "['Figure']", "['Table']", "multimodal-t", "multimodal-f"], 'questions': []},
        'multimodal': {'types': ["multimodal-t", "multimodal-f"], 'questions': []},
    }
    
    # Counters to track how many questions of each type we've found
    counts = {cat: 0 for cat in categories}
    
    print(f"Starting extraction from '{input_file}'...")
    print(f"Goal: Collect {limit_per_type} questions for each of the following categories: {list(categories.keys())}\n")

    # --- 2. File Processing ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Check if we have already collected enough questions for all categories
                if all(counts[cat] >= limit_per_type for cat in categories):
                    print(f"All quotas met. Stopping after reading {i} lines.")
                    break

                try:
                    # Each line is a JSON object
                    data = json.loads(line)
                    doc_questions = data.get('questions', [])

                    for question_obj in doc_questions:
                        q_type_str = str(question_obj.get('type', ''))

                        # Iterate through our defined categories to see if this question matches
                        for cat_name, cat_info in categories.items():
                            # Check if we still need questions for this category
                            if counts[cat_name] < limit_per_type:
                                # Check if the question's type matches one of the types for this category
                                if q_type_str in cat_info['types']:
                                    # Add the full original question object to our list
                                    cat_info['questions'].append(question_obj)
                                    question_obj["doc_name"] = data.get("doc_name")
                                    question_obj["page_indices"] = data.get("page_indices")
                                    counts[cat_name] += 1
                                    # Once a question is categorized, we don't need to check other categories
                                    break 
                
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {i+1}. Skipping.")
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # --- 3. Reporting and Saving ---
    print("\n--- Extraction Summary ---")
    total_extracted = 0
    for cat_name, count in counts.items():
        print(f"Collected {count} questions for category '{cat_name}'.")
        total_extracted += count
    print(f"Total questions extracted: {total_extracted}")

    # Create an output directory if it doesn't exist
    output_dir = "dataset/extracted_questions"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to the '{output_dir}' directory...")

    for cat_name, cat_info in categories.items():
        output_filename = os.path.join(output_dir, f"{cat_name}_{len(cat_info['questions'])}_questions.jsonl")
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            for question_item in cat_info['questions']:
                # Write each question object as a new line in the output file
                out_f.write(json.dumps(question_item) + '\n')
        print(f"Successfully saved to '{output_filename}'")

# --- Main execution block ---
if __name__ == "__main__":
    # Name of your input file.
    # Make sure this file is in the same directory as the script, or provide a full path.
    jsonl_file = 'dataset/MMDocIR_annotations.jsonl' 
    
    # Run the extraction function
    extract_questions_by_type(input_file=jsonl_file, limit_per_type=100)
