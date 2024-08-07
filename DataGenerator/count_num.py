def count_num(data_path):
    item_num = 0
    with open(data_path, 'r') as file_object:
        lines = file_object.readlines()
        print(f"sequence_num={len(lines)}.")
        for line in lines:
            line = line.strip().split('\t')
            item_num += (len(line) - 1)
        print(f"interaction_num={item_num}.")
    return


count_num("../Data/Hamazon_B/train_data.txt")
