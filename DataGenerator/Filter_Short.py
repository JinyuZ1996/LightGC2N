def get_data(data_path):
    out_file = open("../Data_Generator/Video_F.txt", "w")
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        for line in lines:
            temp_sequence = line
            line = line.strip().split('\t')
            if len(line) < 4:
                continue
            else:
                out_file.writelines(temp_sequence)
    return


get_data("../Data_Generator/Video.txt")
