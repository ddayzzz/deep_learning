# coding=utf-8
import tensorflow as tf
from data_model import StockDataSet
from os import sep

class RNNConfig():
    """
    定义 RNN 的一些配置文件
    """
    input_size = 1  # 输入的样本个数
    num_steps = 30  # 展开 RNN 的网络层数(也就是 堆叠的 LSTM的层数)
    lstm_size = 128  # 一个 LSTM 层数
    num_layers = 1
    keep_prob = 0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 50


config = RNNConfig()


def _create_one_cell():
    # 如果需要 dropput 可以添加
    """
    if config.keep_prob < 1.0:
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

    """
    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)


with tf.device('/gpu:0'):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        # 定义输入
        inputs = tf.placeholder(dtype=tf.float32,
                                shape=[None, config.num_steps, config.input_size])
        targets = tf.placeholder(dtype=tf.float32,
                                shape=[None, config.input_size])
        learning_rate = tf.placeholder(tf.float32, None)
        # 定义 RNN 展开后的(堆叠LSTM数量)
        cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(config.num_layers)], state_is_tuple=True) if config.num_layers > 1 else _create_one_cell()
        val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)  # val 的维度(batch_size, num_steps, lstm_size)
        # 调换 val 的维度
        val = tf.transpose(val, [1, 0, 2])  # 维度变为 (num_steps, batch_size, lstm_size)
        last = tf.gather(val, int(val.get_shape()[0]) -1, name='last_lstm_output')
        # 预测
        weight = tf.Variable(initial_value=tf.truncated_normal([config.lstm_size, config.input_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]))
        prediction = tf.matmul(last, weight) + bias
        # 损失
        loss = tf.reduce_mean(tf.square(prediction - targets))
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)
    # 训练过程
    with tf.Session(graph=graph) as sess:
        # merged_summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('log', sess.graph)
        # writer.add_graph(sess.graph)
        # 初始化各种变量
        tf.global_variables_initializer().run(session=sess)
        learning_rates_to_use = [
            config.init_learning_rate * (
                    config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
            ) for i in range(config.max_epoch)]
        # 进行一次 epoch
        stock_dataset = StockDataSet(stock_sym='IBM')
        for epoch_step in range(config.max_epoch):
            current_lr = learning_rates_to_use[epoch_step]

            # Check https://github.com/lilianweng/stock-rnn/blob/master/data_wrapper.py
            # if you are curious to know what is StockDataSet and how generate_one_epoch()
            # is implemented.
            for batch_X, batch_y in stock_dataset.generate_one_epoch(config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr
                }
                train_loss, _ = sess.run([loss, minimize], train_data_feed)
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, sep.join(('model', 'stock_rnn1.ckpt')), global_step=config.max_epoch)
