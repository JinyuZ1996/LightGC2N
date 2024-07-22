# coding: utf-8

from time import time
import logging
from SAGC2N.SA_train import *
from SAGC2N.SA_model import *
import warnings

warnings.filterwarnings("ignore")

param = ParamConf()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    item_dict = load_dict(dict_path=param.item_path)
    user_dict = load_dict(dict_path=param.user_path)
    logging.info("Dictionaries initialized. Loading data...")

    train_data = config_input(data_path=param.train_path, item_dict=item_dict, user_dict=user_dict)
    test_data = config_input(data_path=param.test_path, item_dict=item_dict, user_dict=user_dict)
    if param.fast_running:
        train_data = train_data[:int(param.fast_ratio * len(train_data))]
        logging.info("Data initialized (Fast Running). Transforming to Matrix-form...")
    else:
        logging.info("Data initialized. Transforming to Matrix-form...")

    input_matrices = trans_matrix_form(train_data)
    logging.info("Transformation completed. Generating sequential graph...")

    laplace_list = graph_construction(input_matrices, item_dict, user_dict)
    logging.info("Graph Initialized. Generating batches...")

    train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size, padding_num=len(item_dict),
                                     is_train=True)
    test_batches = generate_batches(input_data=test_data, batch_size=param.batch_size, padding_num=len(item_dict),
                                    is_train=False)
    logging.info("Batches loaded. Initializing Backup_2024_03_01 network...")

    num_items = len(item_dict)
    num_users = len(user_dict)
    module = CapSA(num_items=num_items, num_users=num_users, laplace_list=laplace_list)
    logging.info("Model Initialized. Start training...")

    with tf.Session(graph=module.graph, config=module.config) as sess:
        module.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        best_score = -1
        for epoch in range(param.num_epochs):
            time_start = time()
            loss_train = train_module(sess=sess, module=module, batches_train=train_batches,
                                      learning_rate=param.learning_rate, dropout_rate=param.dropout_rate)
            time_consumption = time() - time_start
            epoch_num = epoch + 1
            logging.info(
                'Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch_num, loss_train, time_consumption))
            if epoch_num % param.eval_verbose == 0:
                test_start = time()
                [rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20] = \
                    model_evaluation(sess=sess, module=module, batches_test=test_batches, eval_length=len(test_data))
                test_consumption = time() - test_start
                logging.info(
                    "Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, "
                    "MRR20 = %.4f" % (epoch_num, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))
                logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch_num, test_consumption))
                if rc_5 >= best_score:
                    best_score = rc_5
                    saver.save(sess, param.check_points, global_step=epoch_num, write_meta_graph=False)
                    logging.info("Recommender performs better, saving current model....")

            train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                             padding_num=len(item_dict), is_train=True)

        logging.info("Recommender training finished.")
    logging.info("All process finished.")
