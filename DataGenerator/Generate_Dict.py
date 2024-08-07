import pandas as pd


# df = pd.read_csv("../Data_Generator/user_dict.txt", sep='\t', header=None)
# df.to_csv('../Data_Generator/User_Dict.csv', index=False, sep=',', encoding='utf-8', header=None)

# df = pd.read_csv("../Data_Generator/Movie_Dict.csv", sep=',', header=None)
# df.to_csv('../Data_Generator/Movie_Dict.txt', index=False, sep='\t', encoding='utf-8', header=None)

def get_data(data_path, dict_path_A, dict_path_B, dict_path_U):
    obj_A = open(dict_path_A, "w")
    obj_B = open(dict_path_B, "w")
    obj_U = open(dict_path_U, "w")
    count_A, count_B, count_U = 0, 0, 0
    user_set, item_A, item_B = [], [], []
    with open(data_path, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            line = line.strip().split('\t')
            sequence_A = ""
            user = line[0]
            if user not in user_set:
                user_set.append(user)
            for item in line[1:]:
                item_info = item.split('|')
                item_id = str(item_info[0])
                if item_id[0] is 'Hvideo_E' and item_id not in item_A:
                    item_A.append(item_id)
                elif item_id[0] is 'Hvideo_V' and item_id not in item_B:
                    item_B.append(item_id)
            # sequence_A.append("\n")
            # sequence_B.append("\n")
            # obj_A.write(sequence_A + "\n")
            # B_object.write(sequence_B + "\n")
            # temp_sequence.append(sequence_all)  # [0]
            # mixed_data.append(temp_sequence)
        # print(user)
        # print(len(user))
        # print(item_A)
        # print(len(item_A))
        # print(item_B)
        # print(len(item_B))
        # for i in range(len(user_set)):
        #     obj_U.write(str(i) + '\t' + user_set[i] + '\n')
        #
        # for j in range(len(item_A)):
        #     obj_A.write(str(j) + '\t' + item_A[j] + '\n')
        #
        # for k in range(len(item_B)):
        #     obj_B.write(str(k) + '\t' + item_B[k] + '\n')
        #
        for index, user in enumerate(user_set):
            obj_U.write(f"{index}\t{user}\n")
        for index, item in enumerate(item_A):
            obj_A.write(f"{index}\t{item}\n")
        for index, item in enumerate(item_B):
            obj_B.write(f"{index}\t{item}\n")
        obj_U.close()
        obj_A.close()
        obj_B.close()
    return

get_data("Hvideo_origin/new_alldata.txt", "Hvideo_origin/A_dict.txt", "Hvideo_origin/B_dict.txt",
         "Hvideo_origin/U_dict.txt")