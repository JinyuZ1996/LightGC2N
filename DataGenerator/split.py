import random

dataset_path = 'Hamazon/M/alldata_M.txt'
with open(dataset_path, 'r') as file:
    lines = file.readlines()
random.shuffle(lines)
split_point = int(len(lines) * 0.8)
train_set = lines[:split_point]
test_set = lines[split_point:]
with open('Hamazon/M/train_data.txt', 'w') as file:
    file.writelines(train_set)
with open('Hamazon/M/test_data.txt', 'w') as file:
    file.writelines(test_set)
print(f"Train_size: {len(train_set)}")
print(f"Test_size: {len(test_set)}")