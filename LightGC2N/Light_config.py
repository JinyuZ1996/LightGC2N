# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

np.seterr(all='ignore')


class ParamConf:
    def __init__(self, dataset="Hvideo_E"):  # Hvideo_E, Hvideo_V, Hamazon_M, Hamazon_B

        # Model parameters
        self.embedding_size = 16
        self.num_layers = 3
        self.num_folded = self.embedding_size
        self.alpha = 3  # the number of latent users
        self.beta = 0.4  # the temperature of SSL loss
        self.gamma = 0.3  # the participation of SSL loss

        # Training parameters
        self.learning_rate = 0.005
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 200
        self.eval_verbose = 10
        self.fast_running = False  # set this "True" for fast-running
        self.fast_ratio = 0.2
        self.gpu_index = '0'

        # data_path
        self.dataset = dataset
        self.train_path = f"../Data/{self.dataset}/train_data.txt"
        self.test_path = f"../Data/{self.dataset}/test_data.txt"
        self.item_path = f"../Data/{self.dataset}/item_dict.txt"
        self.user_path = f"../Data/{self.dataset}/user_dict.txt"
        self.check_points = f"../check_points/{self.dataset}.ckpt"


def load_dict(dict_path):
    dict_output = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for dict_element in elements:
        # Strip the line of leading and trailing whitespace and split it by the tab delimiter
        dict_element = dict_element.strip().split('\t')
        # Assign the value to the key in the dictionary
        dict_output[dict_element[1]] = int(dict_element[0])
    return dict_output


def config_input(data_path, item_dict, user_dict):
    """
        Configure input data for the model.
        This function reads the data from the specified file and prepares it in a format suitable for training or inference.
    """
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            # Clean the data and split it by tabs
            elements = line.strip().split('\t')

            # Initialize lists for user sequence and positions
            user_sequence, positions = [], []

            # Build the user sequence and positions
            for index, element in enumerate(elements[1:], start=1):
                user_sequence.append(item_dict[element])
                positions.append(index - 1)

            # Add the user ID to the beginning of the sequence
            user_sequence.insert(0, user_dict[elements[0]])

            # Calculate sequence length (excluding the user ID)
            sequence_length = len(user_sequence) - 1

            # Get the target item ID
            target_item_id = item_dict[elements[-1]]

            data.append([user_sequence, positions, sequence_length, target_item_id])

    return data


def matrix2list(matrix):
    df = pd.DataFrame(matrix, columns=['row', 'column'])
    unique_df = df.drop_duplicates()
    return unique_df.values.tolist()


def trans_matrix_form(data):
    """
    Transform input data into matrix form.
    This function processes a list of records, where each record contains a user sequence,
    and constructs four matrices that represent sequential and interactive relationships
    between users and items.
    """
    matrix_i2i, matrix_i2u, matrix_u2i, matrix_u2u = [], [], [], []

    for record in data:
        sequence = record[0]
        user = int(sequence[0])  # Extract the user ID from the sequence
        items = [int(i) for i in sequence[1:]]  # Extract the interacted item IDs

        # Construct the sequential relations of items within a user's sequence
        for item_index in range(len(items) - 1):
            item_temp = items[item_index]
            next_item = items[item_index + 1]
            matrix_i2i.append([item_temp, item_temp])  # Self-relation of items
            matrix_i2i.append([item_temp, next_item])  # Relation from one item to the next

        # Construct the interactive relations between the user and all items
        for item in items:
            matrix_u2i.append([user, item])  # User-item interaction
            matrix_i2u.append([item, user])  # Item-user interaction

        # User's self-relation
        matrix_u2u.append([user, user])

    # Convert the lists of tuples into NumPy arrays
    matrix_i2i = np.array(matrix2list(matrix_i2i))
    matrix_i2u = np.array(matrix2list(matrix_i2u))
    matrix_u2i = np.array(matrix2list(matrix_u2i))
    matrix_u2u = np.array(matrix2list(matrix_u2u))

    return [matrix_i2i, matrix_i2u, matrix_u2i, matrix_u2u]


def matrix2inverse(array_in, row_index, col_index, matrix_dimension):
    # Extract the rows and columns from the input matrix, adjusting for the given index offsets
    matrix_rows = [row + row_index for row, _ in array_in]
    matrix_columns = [col + col_index for _, col in array_in]

    # Initialize a list to hold the values for the inverse matrix, all set to 1
    matrix_value = [1.] * len(matrix_rows)

    # Create the sparse matrix using the Scipy COO format, which is efficient for large matrices
    inverse_matrix = coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                shape=(matrix_dimension, matrix_dimension))
    return inverse_matrix


def graph_construction(matrices, item_dict, user_dict):
    """
    Construct a graph from matrices.
    This function takes in a list of matrices representing various relationships
    and constructs a graph by inverting the matrices to create adjacency matrices.
    It then converts these matrices into a sparse format for efficient memory usage.
    """
    graph_matrices = []  # Initialize a list to store the constructed graph matrices
    item_size = len(item_dict)  # Get the number of unique items
    user_size = len(user_dict)  # Get the number of unique users
    num_all = item_size + user_size  # Calculate the total number of entities

    # Create the inverse matrices for the given relationships
    matrix_i2i = matrix2inverse(matrices[0], row_index=0, col_index=0, matrix_dimension=num_all)
    matrix_i2u = matrix2inverse(matrices[1], row_index=0, col_index=item_size, matrix_dimension=num_all)
    matrix_u2i = matrix2inverse(matrices[2], row_index=item_size, col_index=0, matrix_dimension=num_all)
    matrix_u2u = matrix2inverse(matrices[3], row_index=item_size, col_index=item_size, matrix_dimension=num_all)

    # Append the inverse matrices to the graph matrices list
    graph_matrices.append(matrix_i2i)
    graph_matrices.append(matrix_i2u)
    graph_matrices.append(matrix_u2i)
    graph_matrices.append(matrix_u2u)

    # Convert the graph matrices into a sparse matrix format (COO)
    laplace_list = [adj.tocoo() for adj in graph_matrices]

    # Sum all the sparse matrices to create the overall graph Laplacian matrix
    return sum(laplace_list)


def load_batches(batch, padding_num):
    user, sequence, position, length, target = [], [], [], [], []
    # Determine the maximum sequence length
    max_length = max([data_index[2] for data_index in batch])

    for data_index in batch:
        # Extract user ID, sequence, positions, length, and target item from the data index
        user.append(data_index[0][0])
        sequence.append(data_index[0][1:] + [padding_num] * (max_length - data_index[2]))
        position.append(data_index[1][0:] + [padding_num] * (max_length - data_index[2]))
        length.append(data_index[2])
        target.append(data_index[3])

    return np.array(user), np.array(sequence), np.array(position), np.array(length).reshape(-1, 1), np.array(target)


def generate_batches(input_data, batch_size, padding_num, is_train):
    user_all, sequence_all, position_all, length_all, target_all = [], [], [], [], []
    # Calculate the number of batches
    num_batches = int(len(input_data) / batch_size)

    if is_train:
        # Shuffle the input data if it is for training
        np.random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        user, sequence, position, length, target = load_batches(batch, padding_num)
        user_all.append(user)
        sequence_all.append(sequence)
        position_all.append(position)
        length_all.append(length)
        target_all.append(target)

    return user_all, sequence_all, position_all, length_all, target_all, num_batches
