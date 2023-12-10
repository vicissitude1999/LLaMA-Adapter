import json

def create_json_object(question_id, model_id, response):
    return {"question_id": question_id, "model_id": model_id, "choices": [{"index": 0, "turns": [response]}]}

def create_jsonl_file(data, output_file):
    with open(output_file, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

# Example usage
responses = [
    "r1", "r2"
]

def format_responses(model_id, responses, output_file):
    # Convert responses to the desired format
    formatted_data = [
        create_json_object(i+1, model_id, response) for i, response in enumerate(responses)
    ]

    # Output to JSONL file
    create_jsonl_file(formatted_data, output_file)

def read_questions(input_file):
    def extract_text_from_json(json_line):
        return json_line["turns"][0] if "turns" in json_line else None

    questions = []
    # Open the file and process each line
    with open(input_file, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            question = extract_text_from_json(json_data)
            questions.append(question)
    return questions

if __name__ == "__main__":
    format_responses("model1", responses, 'output.jsonl')