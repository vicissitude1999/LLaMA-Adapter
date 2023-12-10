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
    
    train_data = []
    test_data = []
    
    indices = np.random.permutation(len(item_subset))
    train_indices = indices[:50000]
    test_indices = indices[50000:51000]
    
    for idx in train_indices:
        train_data.append(item_subset[idx])
    for idx in test_indices:
        test_data.append(item_subset[idx])
    
    with open('math_data_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open('math_data_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)


def combine_alpaca_math():
    with open('alpaca_data.json', 'r') as f:
        dalpaca = json.load(f)
    
    with open('math_data_train.json', 'r') as f:
        dmath = json.load(f)
    
    dcombined = dalpaca + dmath
    with open('alpaca_math_data.json', 'w') as f:
        json.dump(dcombined, f, indent=4)

if __name__ == "__main__":
    main()
    # combine_alpaca_math()