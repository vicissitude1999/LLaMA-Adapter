import json
import numpy as np
import sympy

def create_json_object(question_id, model_id, response):
    return {"question_id": question_id, "model_id": model_id, "choices": [{"index": 0, "turns": [response]}]}

def create_jsonl_file(data, output_file):
    with open(output_file, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')


def format_responses(model_id, responses, output_file):
    # Convert responses to the desired format
    formatted_data = [
        create_json_object(i, model_id, responses[i]) for i in responses
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

def read_math(input_file):
    instructs = []
    
    with open(input_file, 'r') as f:
        d = json.load(f)
        for item in d:
            instructs.append(item["instruction"])
    
    return instructs
    
def compute_error(path_response_pred, path_response_true):
    responses_pred = []
    
    with open(path_response_pred, 'r') as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            
            response = item["choices"][0]["turns"][0].split("Response:")[-1]
            try:
                response = sympy.Rational(response)
                response = float(response)
            except:
                response = 0
            responses_pred.append(response)
    
    # print(responses_pred)
    
    responses_true = []
    
    with open(path_response_true, 'r') as f:
        d = json.load(f)
        for i in range(len(responses_pred)):
            item = d[i]
            
            response = item["output"]
            response = sympy.Rational(response)
            response = float(response)
            responses_true.append(response)
    
    # print(responses_true)
    
    responses_pred = np.array(responses_pred)
    responses_true = np.array(responses_true)
    diff = np.abs(responses_pred - responses_true)
    print("Accuracy", (diff < 1e-7).sum() / len(diff))
    
    print(f"min {np.min(diff)} 25% {np.percentile(diff, 25)} 50% {np.percentile(diff, 50)} 75% {np.percentile(diff, 75)} max {np.max(diff)}")
    
if __name__ == "__main__":
    compute_error("adapter_adapter_len10_layer30_epoch5.jsonl",
                  "data/math_data_test.json")