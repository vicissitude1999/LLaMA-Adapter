import json
import numpy as np

features = ["solve", "Solve", "what is", "What is", "Given", "given", "find", "Find"]


def main():
    item_subset = []
    
    counter = {}
    for feat in features:
        counter[feat] = 0
        
    with open('math_data.json', 'r') as f:
        d = json.load(f)
        for item in d:
            ins = item['instruction']
            
            for feat in features:
                if feat in ins:
                    counter[feat] += 1
                    item_subset.append(item)
                    break
    print(counter)
    
    item_subset_small = []
    indices = np.random.choice(len(item_subset), 20000)
    for idx in indices:
        item_subset_small.append(item_subset[idx])
    
    with open('math_data_subset.json', 'w') as f:
        json.dump(item_subset_small, f, indent=4)


def combine_alpaca_math():
    with open('alpaca_data.json', 'r') as f:
        dalpaca = json.load(f)
    
    with open('math_data_subset.json', 'r') as f:
        dmath = json.load(f)
    
    dcombined = dalpaca + dmath
    with open('alpaca_math_data.json', 'w') as f:
        json.dump(dcombined, f, indent=4)

if __name__ == "__main__":
    main()
    combine_alpaca_math()