import argparse
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default="tf")

cfg = parser.parse_args()

if not os.path.exists("data.pkl"):
    with open("data.pkl", 'wb') as f:
        data = []
        for i in range(5):
            A = np.random.randn(1, 2, 2, 3)
            B = np.random.randn(1, 2, 2, 3)
            data.append((A, B))
        pickle.dump(data, f)


with open('data.pkl', 'rb') as f:
    data = pickle.load(f)


if cfg.type == "tf":
    import tensorflow as tf
    from config import config
    from CX.CX_helper import CX_loss_helper

    with tf.Session() as sess:
        for d in data:
            A, B = d
            A = tf.convert_to_tensor(A, dtype=tf.float32)
            B = tf.convert_to_tensor(B, dtype=tf.float32)
            loss = CX_loss_helper(A, B, config.CX)
            l = sess.run(loss)
            print("loss  ======  " + str(l))
            # break
elif cfg.type == "torch":
    import torch
    from model.generator import CXLoss
    for d in data:
        A, B = d
        A = np.transpose(A, (0, 3, 1, 2))
        B = np.transpose(B, (0, 3, 1, 2))
        print(A.shape)
        print(B.shape)
        A = torch.from_numpy(A).type(torch.FloatTensor)
        B = torch.from_numpy(B).type(torch.FloatTensor)
        loss_type = CXLoss(sigma=0.5)
        loss = loss_type(A, B)
        print("loss  ======  " + str(loss.item()))
        # break
