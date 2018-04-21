#! /usr/bin/env python3.4

import datetime
import operator
import os
import pdb
import lstm_cnn.tensorflow.qaData as qaData

import tensorflow as tf
# from cnn.tensorflow import insurance_qa_data_helpers
from lstm_cnn.tensorflow.insqa_lstm_cnn import InsQALSTMCNN
from lstm_cnn.tensorflow.insurance_qa_data_helpers import read_raw, read_alist_answers, build_vocab, load_test, \
    load_data_val_6, load_data_6



# Model Hyperparameters
embedding_dim = 50
filter_sizes = "1,2,3,5"
num_filters = 500
# Training parameters
seq_length = 100
batch_size = 100
num_steps = 5000000
evaluate_every = 6000
checkpoint_every = 6000
# Misc Parameters
allow_soft_placement = True
log_device_placement = False
# tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
#
# # Training parameters
# tf.flags.DEFINE_integer("seq_length", 100, "Sequence Length (default: 200)")
# tf.flags.DEFINE_integer("batch_size", 20, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_steps", 5000000, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 6000, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 6000, "Save model after this many steps (default: 100)")
# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


# Data Preparatopn
# ==================================================

#train_file = '../../insuranceQA/train'
train_file = '/Users/cjf/Downloads/C题部分数据/train_data_sample.json'
#dev_file = '../../insuranceQA/test1'
dev_file = '/Users/cjf/Downloads/C题部分数据/dev_data_sample.json'
precision = '../../insuranceQA/test1.acc'
embedding_file = '/Users/cjf/Downloads/zhwiki/zhwiki_2017_03.sg_50d.word2vec'
resultFile = "predictRst.score"

def evaluate_model(model, session):
    #testList = load_test(val_file)

    scoreList = []
    i = int(0)
    while True:
        x_test_1, x_test_2 = qaData.load_data_val_6(qTest, aTest, i,
                                                       batch_size)
        batch_scores = model.dev_step(x_test_1, x_test_2, session)
        for score in batch_scores[0]:
            scoreList.append(score)
        i += batch_size
        if i >= len(qTest):
            break
    # for question, answer in qaData.testingBatchIter(qTest, aTest, batch_size):
    #     batch_scores = model.dev_step(question, answer, session)
    #     for score in batch_scores[0]:
    #         scoreList.append(score)
    with open(resultFile, 'w') as file:
        for score in scoreList:
            file.write("%.9f" % score + '\n')
    sessdict = {}
    index = int(0)
    # for line in open(dev_file):
    #     items = line.strip().split(' ')
    #     qid = items[1].split(':')[1]
    #     if not qid in sessdict:
    #         sessdict[qid] = []
    #     sessdict[qid].append((scoreList[index], items[0]))
    #     index += 1
    #     if index >= len(testList):
    #         break
    # lev1 = float(0)
    # lev0 = float(0)
    # of = open(precision, 'a')
    # for k, v in sessdict.items():
    #     v.sort(key=operator.itemgetter(0), reverse=True)
    #     score, flag = v[0]
    #     if flag == '1':
    #         lev1 += 1
    #     if flag == '0':
    #         lev0 += 1
    # of.write('lev1:' + str(lev1) + '\n')
    # of.write('lev0:' + str(lev0) + '\n')
    # print('lev1 ' + str(lev1))
    # print('lev0 ' + str(lev0))
    # print("top-1 accuracy: %s" % (lev1 * 1.0 / (lev1 + lev0)))
    # of.close()


def create_model(session):
    model = InsQALSTMCNN(
        sequence_length=seq_length,
        batch_size=batch_size,
        embeddings=embedding,
        embedding_size=embedding_dim,
        filter_sizes=list(map(int, filter_sizes.split(","))),
        num_filters=num_filters)
    print("Writing to {}\n".format(out_dir))

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if True:
        #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, "runs/checkpoints-41999")
    else:
        print("model file not loaded correctly. Start fresh new model")
        # Initialize all variables
        session.run(tf.global_variables_initializer())
    return model


# Training
# ==================================================
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
# Checkpoint directory. tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load data
# print("Loading data...")
#
# vocab = build_vocab(train_file, dev_file)
# vocab_size = len(vocab)
# alist_answers = read_alist_answers(train_file)
# raw = read_raw(train_file)
#
# print("Load done...")

session_conf = tf.ConfigProto(
    allow_soft_placement=allow_soft_placement,
    log_device_placement=log_device_placement,
    # device_count={'GPU': 0} uncomment if you have a GPU but don't want to use it
)
session_conf.gpu_options.allow_growth = True
# 读取测试数据
print("正在读取embedding文件")
embedding, word2idx = qaData.loadEmbedding(embedding_file)

with tf.Session(config=session_conf) as sess:
    model = create_model(sess)

    print("start...")
    #准备训练数据
    print("正在准备训练数据，大约需要五分钟...")
    qTrain, aTrain, lTrain, qIdTrain, aIdTrain = qaData.loadjsonData(train_file, word2idx, seq_length, True)
    tqs, tta, tfa = [], [], []
    question, trueAnswer, falseAnswer = qaData.trainingBatchIter(qTrain, aTrain,
                                                                lTrain, qIdTrain,
                                                                batch_size)


    # question, trueAnswer, falseAnswer = load_data_6(vocab, alist_answers, raw,
    #                                               batch_size)
    # print(question)

    # 读取测试数据
    print("正在载入测试数据，大约需要一分钟...")
    qTest, aTest, _, qIdTest, aIdTest = qaData.loadjsonData(dev_file, word2idx, seq_length)
    print("测试数据加载完成")

    evaluate_model(model, sess)
    # Generate batches
    # Training loop. For each batch...
    index = 0
    for n_step in range(num_steps):
        try:
            train_question, train_trueAnswer, train_falseAnswer = qaData.load_data_6(question, trueAnswer, falseAnswer,
                                                           batch_size,index)

            index += batch_size
            if index > len(question): index = 0
            #for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
            model.train_step(train_question, train_trueAnswer, train_falseAnswer, sess, n_step)
            if (n_step + 1) % checkpoint_every == 0:
                path = model.saver.save(sess, checkpoint_dir, global_step=n_step)
                model.saver.save(sess,"newmodel")
                print("Saved model checkpoint to {}\n".format(path))
            if (n_step + 1) % evaluate_every == 0:
                print("\nEvaluation:")
                evaluate_model(model, sess)
                print("")

        except Exception as e:
            print(e)
