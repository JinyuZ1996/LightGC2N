# coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras import layers
from Light_config import *
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2024
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
param = ParamConf()


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_inputs():
    user = tf.placeholder(dtype=tf.int32, shape=[None, ], name='user')
    sequence = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence')
    position = tf.placeholder(dtype=tf.int32, shape=[None, None], name="position")
    length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='length')
    target = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
    return user, sequence, position, length, target, learning_rate, dropout_rate


def optimizer(loss, learning_rate):
    basic_op = tf.train.AdamOptimizer(learning_rate)
    gradients = basic_op.compute_gradients(loss)
    # Clip the gradients by value between -5 and 5
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    model_op = basic_op.apply_gradients(capped_gradients)
    return model_op


def cal_loss_mean(target, pred):
    # cross_entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=pred)
    loss_mean = tf.reduce_mean(loss, name='loss_mean')

    return loss_mean


def squash(s, axis=-1, epsilon=1e-7, name=None):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector


def linear_attention(input_features, num_capsules, dim_capsule):
    query, value = input_features, input_features
    key = tf.transpose(input_features, perm=[1, 0])
    attention_maps = tf.matmul(key, value)
    normed_ebd = tf.nn.softmax(attention_maps, axis=1)
    scored_ebd = tf.matmul(query, normed_ebd)
    ebd_dropout = tf.keras.layers.Dropout(rate=param.dropout_rate)(scored_ebd)
    map_to_caps = layers.Dense(units=num_capsules * dim_capsule, activation=None)
    caps_ebd = map_to_caps(ebd_dropout)

    return caps_ebd


def dynamic_routing(b_ij, inputs, num_outputs, routing_iterations):
    global v_j
    assert num_outputs == b_ij.get_shape().as_list()[1]
    for iteration in range(routing_iterations):
        c_ij = tf.nn.softmax(b_ij, dim=2)
        s_j = tf.reduce_sum(c_ij * inputs, axis=1, keepdims=True)
        v_j = squash(s_j)
        if iteration < routing_iterations - 1:
            b_ij += tf.reduce_sum(inputs * v_j, axis=0, keepdims=True)
    return v_j


class CapsuleLayer:
    def __init__(self, num_capsules, dim_capsule, routings=0, layer_type='CONV'):
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.layer_type = layer_type

    def __call__(self, input):
        if self.layer_type == 'ln_att':
            # Linear Attention
            capsules = linear_attention(input, self.num_capsules, self.dim_capsule)
            capsules = tf.reshape(capsules, (input.get_shape()[0], self.num_capsules, self.dim_capsule))
            capsules = squash(capsules)
            return capsules
        elif self.layer_type == 'CONV':
            # Point-wise Conv1D
            conv_input = tf.expand_dims(input, axis=1)
            capsules = layers.Conv1D(self.num_capsules * self.dim_capsule, kernel_size=1, strides=1,
                                     padding='valid')(conv_input)
            capsules = tf.reshape(capsules, (input.get_shape()[0], self.num_capsules, self.dim_capsule))
            capsules = squash(capsules)
            return capsules
        elif self.layer_type == 'Att_Conv':
            # Conv1D + Linear Attention
            conv_input = tf.expand_dims(input, axis=1)
            capsules_conv = layers.Conv1D(self.num_capsules * self.dim_capsule, kernel_size=1, strides=1,
                                          padding='valid')(conv_input)
            capsules_att = linear_attention(input, self.num_capsules, self.dim_capsule)
            capsules_att = tf.expand_dims(capsules_att, axis=1)
            capsules = tf.add(capsules_att, capsules_conv)
            capsules = tf.contrib.layers.layer_norm(capsules)
            capsules = tf.reshape(capsules, (input.get_shape()[0], self.num_capsules, self.dim_capsule))
            capsules = squash(capsules)
            return capsules
        elif self.layer_type == 'MDR':
            # Merge Dynamic Routing
            inputs_shape = input.get_shape()
            num_inputs = inputs_shape[-2].value  # get the capsule_num of previous layer
            inputs = tf.reshape(input, shape=(-1, num_inputs, self.dim_capsule))
            b_ij = tf.constant(np.zeros([inputs_shape[0], num_inputs, self.dim_capsule], dtype=np.float32))
            capsules = dynamic_routing(b_ij, inputs, self.num_capsules, self.routings)

            return capsules


class CapSA:
    def __init__(self, num_items, num_users, laplace_list):
        os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.num_users = num_users
        self.laplace_list = laplace_list

        self.num_latents = param.alpha
        self.temperature = param.beta
        self.batch_size = param.batch_size

        self.ebd_size = param.embedding_size
        self.num_layers = param.num_layers
        self.num_folded = param.num_folded
        self.gamma = param.gamma

        self.is_train = True

        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.user, self.sequence, self.position, self.length, self.target, self.lr, self.dropout_rate = get_inputs()

            with tf.name_scope('architecture'):
                self.subspace_basis = self.initialize_subspace_basis()
                self.all_weights = self.init_weights()
                self.graph_ebd_items, self.graph_ebd_users = \
                    self.graph_encoder(num_items=num_items, num_users=num_users, graph_matrix=laplace_list)
                self.cluster_ebd, self.basic_seq_ebd = self.deep_subspace_clustering(
                    graph_ebd_items=self.graph_ebd_items)
                self.pred = self.output_prediction(basic_seq_ebd=self.basic_seq_ebd,
                                                   graph_ebd_users=self.graph_ebd_users,
                                                   clustered_seq_ebd=self.cluster_ebd)

            with tf.name_scope('loss'):
                self.loss_mean = cal_loss_mean(self.target, self.pred)
                self.loss_ssl = self.cal_loss_ssl(self.basic_seq_ebd, self.cluster_ebd)
                self.loss_total = self.loss_mean + self.gamma * self.loss_ssl

            with tf.name_scope('optimizer'):
                self.model_op = optimizer(self.loss_total, self.lr)

    def init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.ebd_size]))
        all_weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.ebd_size]))
        return all_weights

    def unzip_laplace(self, laplace_list):
        unzip_info = []
        fold_len = (laplace_list.shape[0]) // self.num_folded
        for i_fold in range(self.num_folded):
            start = i_fold * fold_len
            if i_fold == self.num_folded - 1:
                end = laplace_list.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            unzip_info.append(_convert_sp_mat_to_sp_tensor(laplace_list[start:end]))
        return unzip_info

    def graph_encoder(self, num_items, num_users, graph_matrix):
        # Generate a set of adjacency sub-matrix.
        with tf.variable_scope('graph_encoder', reuse=tf.AUTO_REUSE):
            graph_info = self.unzip_laplace(graph_matrix)
            ego_embeddings = tf.concat([self.all_weights['item_embedding'], self.all_weights['user_embedding']], axis=0)
            all_embeddings = [ego_embeddings]

            for k in range(self.num_layers):
                temp_embed = []
                for f in range(self.num_folded):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(graph_info[f], ego_embeddings))
                node_embeddings = tf.concat(temp_embed, 0)
                temp_item_ebd, temp_user_ebd = tf.split(node_embeddings, [num_items, num_users], 0)
                primary_items = CapsuleLayer(num_capsules=1, dim_capsule=self.ebd_size, routings=0,
                                             layer_type='Att_Conv')(temp_item_ebd)
                primary_caps = CapsuleLayer(num_capsules=self.num_latents, dim_capsule=self.ebd_size, routings=1,
                                            layer_type='CONV')(temp_user_ebd)
                digit_caps = CapsuleLayer(num_capsules=self.num_latents, dim_capsule=self.ebd_size, routings=3,
                                          layer_type='MDR')(primary_caps)
                node_embeddings = tf.concat([digit_caps, primary_items], axis=0)
                node_embeddings = tf.reduce_sum(node_embeddings, axis=1)
                all_embeddings += [node_embeddings]

            all_embeddings = tf.stack(all_embeddings, 1)  # layer-wise aggregation
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
            graph_ebd_items, graph_ebd_users = tf.split(all_embeddings, [num_items, num_users], 0)
        return graph_ebd_items, graph_ebd_users

    def initialize_subspace_basis(self):
        # Initialize subspace basis for contrastive subspace alignment
        init_basis = tf.get_variable(
            'subspace_basis',
            shape=[self.num_latents, self.ebd_size],
            initializer=tf.random_normal_initializer()
        )
        return init_basis

    def refine_basis(self, ebd_in):
        # Compute distances and assignments
        distances = tf.reduce_sum(tf.square(tf.expand_dims(ebd_in, axis=1) - self.subspace_basis), axis=2)
        assignments = tf.argmin(distances, axis=1)

        # refine subspace_basis
        means = []
        for user_index in range(self.num_latents):
            # Gather the embeddings of users assigned to this subspace
            mask = tf.equal(assignments, user_index)
            masked_ebds = tf.boolean_mask(ebd_in, mask)
            # Calculate mean of the embeddings
            mean = tf.reduce_mean(masked_ebds, axis=0)
            means.append(mean)

        refined_subspace_basis = tf.stack(means, axis=0)

        # Assign the refined subspace_basis
        update_op = tf.assign(self.subspace_basis, refined_subspace_basis)
        return update_op

    def deep_subspace_clustering(self, graph_ebd_items):
        with tf.variable_scope('s_clustering', reuse=tf.AUTO_REUSE):
            basic_seq_ebd = tf.nn.embedding_lookup(graph_ebd_items, self.sequence)
            basic_seq_ebd = tf.reduce_max(basic_seq_ebd, 1)

            # Refine the subspace_basis
            self.refine_basis(basic_seq_ebd)  # (num_latents, 16)
            # Use the refined subspace_basis to find the closest subspace for each user embedding
            distances = tf.sqrt(
                tf.reduce_sum(tf.square(tf.expand_dims(basic_seq_ebd, axis=1) - self.subspace_basis),
                              axis=2))  # (?, num_latents)
            clusters = tf.argmin(distances, axis=1)  # (?,)
            cluster_ebd = tf.gather(self.subspace_basis, clusters)  # (?, 16)

        return cluster_ebd, basic_seq_ebd

    def output_prediction(self, basic_seq_ebd, graph_ebd_users, clustered_seq_ebd):
        with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
            user_ebd = tf.nn.embedding_lookup(graph_ebd_users, self.user)
            concat_ebd = tf.concat([basic_seq_ebd, clustered_seq_ebd, user_ebd], axis=1)
            concat_ebd = tf.keras.layers.Dropout(rate=self.dropout_rate)(concat_ebd)
            prediction = tf.keras.layers.Dense(units=self.num_items, activation=None,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))(
                concat_ebd)

        return prediction

    def cal_loss_ssl(self, basic_seq_ebd, cluster_ebd):
        normalize_basic_ebd = tf.nn.l2_normalize(basic_seq_ebd, 1)
        normalize_refined_ebd = tf.nn.l2_normalize(cluster_ebd, 1)

        # Compute the similarity matrix between the normalized basic and refined embeddings
        similarity = tf.matmul(normalize_basic_ebd, normalize_refined_ebd, transpose_b=True)

        # Extract the diagonal elements which represent the similarity between the embeddings and themselves
        # Divide by temperature to control the concentration of the distribution
        positive_logits = tf.linalg.diag_part(similarity) / self.temperature
        batch_size = tf.shape(normalize_basic_ebd)[0]
        diagonal_zeros = tf.zeros(batch_size, dtype=tf.float32)
        # Set the diagonal of the similarity matrix to zeros to avoid comparing an embedding with itself
        # Divide by temperature to control the concentration of the distribution
        negative_logits = tf.linalg.set_diag(similarity, diagonal_zeros) / self.temperature

        num_negative = batch_size - 1
        positive_loss = -tf.math.log(tf.nn.sigmoid(positive_logits))
        negative_loss = -tf.reduce_sum(tf.math.log(1 - tf.nn.sigmoid(negative_logits)), axis=1)

        num_positive = tf.cast(1, tf.float32)
        num_negative = tf.cast(num_negative, tf.float32)

        # Compute the final loss
        info_nce_loss = (positive_loss + negative_loss) / (num_positive + num_negative)
        info_nce_loss = tf.reduce_mean(info_nce_loss)

        return info_nce_loss

    def model_training(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position: position, self.length: length,
                     self.target: target,
                     self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = True
        return sess.run([self.loss_total, self.model_op], feed_dict)

    def model_evaluation(self, sess, user, sequence, position, length, target, learning_rate, dropout_rate):

        feed_dict = {self.user: user, self.sequence: sequence, self.position: position, self.length: length,
                     self.target: target, self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = False
        return sess.run(self.pred, feed_dict)
