import os
import time
from datetime import timedelta
import tensorflow as tf

from meetingschedule.lstm.lstm_model import TextLSTM
from meetingschedule.lstm.data_helpers import *
from meetingschedule.lstm.config import TLSTMConfig



tensorboard_dir = "tensorboard/textlstm"
save_dir = 'checkpoints/task_textlstm'
best_save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



def train():
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config = TLSTMConfig()
    model = TextLSTM(config)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)    # 用来显示标量信息
    merged_summary = tf.summary.merge_all()     # 将所有summary全部保存到磁盘，以便tensorboard显示
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()    # 保存模型

    print("loading training and validation data...")
    train_texts, train_labels, test_texts, test_labels = text_ready()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    writer.add_graph(session.graph)

    print("training and evaluation...")
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvment = 1000

    flag = False
    for epoch in range(config.num_epochs):
        print("epoch:", epoch + 1)
        batch_train = batch_iter(train_texts, train_labels, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: config.dropout_keep_prob
            }

            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)

                data_len = len(test_texts)
                batch_eval = batch_iter(test_texts, test_labels, 128)
                total_loss = 0.0
                total_acc = 0.0
                for x_batch, y_batch in batch_eval:
                    batch_len = len(x_batch)
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.keep_prob: 1.0
                    }
                    loss, acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    total_loss += loss * batch_len
                    total_acc += acc * batch_len

                loss_val = total_loss / data_len
                acc_val = total_acc / data_len

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=best_save_path) # save_path: String.  Prefix of filenames created for the checkpoint.
                    improved_str = "*"
                else:
                    improved_str = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvment:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def predict(sentences):
    data = pad_seq(sentences)

    config = TLSTMConfig()
    model = TextLSTM(config)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=best_save_path)
    re = session.run(model.y_pred_cls, feed_dict={model.input_x: data, model.keep_prob: 1.0})
    return re


if __name__ == '__main__':
    train()

    sentences = ['我明天联系你',]
    # print(predict(sentences))
