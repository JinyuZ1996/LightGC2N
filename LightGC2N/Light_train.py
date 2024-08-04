# coding: utf-8

from LightGC2N.Light_config import *
import os

param = ParamConf()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index


def train_module(sess, module, batches_train, learning_rate, dropout_rate):
    # Check if batches_train is a list or tuple
    if not isinstance(batches_train, (list, tuple)) or len(batches_train) != 6:
        raise ValueError("batches_train should be a list or tuple with 6 elements.")

    users, sequences, positions, lengths, targets, batch_num = batches_train

    shuffled_indices = np.random.permutation(batch_num)
    total_loss = 0

    for index in shuffled_indices:
        user = users[index]
        sequence = sequences[index]
        position = positions[index]
        length = lengths[index]
        target = targets[index]

        batch_loss, _ = module.model_training(sess=sess, user=user, sequence=sequence, position=position, length=length,
                                              target=target, learning_rate=learning_rate, dropout_rate=dropout_rate)
        total_loss += batch_loss

    avg_loss = total_loss / batch_num
    return avg_loss


def model_evaluation(sess, module, batches_test, eval_length):
    user_all, sequence_all, position_all, length_all, target_all, test_batch_num = (
        batches_test[0], batches_test[1], batches_test[2],
        batches_test[3], batches_test[4], batches_test[5])
    rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20 = 0, 0, 0, 0, 0, 0
    for batch_index in range(test_batch_num):
        test_user = user_all[batch_index]
        test_sequence = sequence_all[batch_index]
        test_position = position_all[batch_index]
        test_length = length_all[batch_index]
        test_target = target_all[batch_index]
        prediction = module.model_evaluation(sess=sess, user=test_user, sequence=test_sequence, position=test_position,
                                             length=test_length, target=test_target, learning_rate=param.learning_rate,
                                             dropout_rate=0)
        recall, mrr = eval_metrics(prediction, test_target, [5, 10, 20])
        rc_5 += recall[0]
        rc_10 += recall[1]
        rc_20 += recall[2]
        mrr_5 += mrr[0]
        mrr_10 += mrr[1]
        mrr_20 += mrr[2]
    return [rc_5 / eval_length, rc_10 / eval_length, rc_20 / eval_length, mrr_5 / eval_length, mrr_10 / eval_length,
            mrr_20 / eval_length]


def eval_metrics(predictions, ground_truths, k_values):
    Recalls, MRRs = [], []
    sorted_indices = predictions.argsort()
    # Iterate over each k value in the provided options
    for k in k_values:
        Recalls.append(0)
        MRRs.append(0)

        # Get the top-k indices from the sorted indices list
        top_k_indices = sorted_indices[:, -k:]

        current_index = 0
        # Iterate over the ground truth list
        while current_index < len(ground_truths):
            # Find the position of the ground truth in the top-k indices
            match_positions = np.argwhere(top_k_indices[current_index] == ground_truths[current_index])

            # If the ground truth is found in the top-k predictions
            if len(match_positions) > 0:
                # Increment the recall for the current k value
                Recalls[-1] += 1
                # Calculate the MRR contribution for the current prediction
                # and add it to the MRR for the current k value
                MRRs[-1] += 1 / (k - match_positions[0][0])
            else:
                # If the ground truth is not found, no change to recall and MRR
                Recalls[-1] += 0
                MRRs[-1] += 0
            current_index += 1

    return Recalls, MRRs
