#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import random


data = ["we", "build", "neural", "networks"]
word_to_idx = {w: i for i, w in enumerate(data)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(data)

# one-hot encoding
def one_hot(index, size):
    return [1 if i == index else 0 for i in range(size)]

X = [one_hot(word_to_idx[w], vocab_size) for w in data[:3]]
Y = one_hot(word_to_idx[data[3]], vocab_size)


hidden_size = 8
learning_rate = 0.1

Wxh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]
Whh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(hidden_size)]

bh = [0.0 for _ in range(hidden_size)]
by = [0.0 for _ in range(vocab_size)]



def tanh(x):
    return [math.tanh(i) for i in x]

def dtanh(x):
    return [1 - i ** 2 for i in x]

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e = sum(e_x)
    return [i / sum_e for i in e_x]

def vector_add(a, b):
    return [i + j for i, j in zip(a, b)]

def matmul(v, W):
    return [sum(v[i] * W[i][j] for i in range(len(v))) for j in range(len(W[0]))]


for epoch in range(1000):
    h = [0.0 for _ in range(hidden_size)]
    hs = []

    for x_t in X:
        h_input = vector_add(matmul(x_t, Wxh), matmul(h, Whh))
        h = tanh(vector_add(h_input, bh))
        hs.append(h)

    y_raw = vector_add(matmul(h, Why), by)
    y_pred = softmax(y_raw)

    loss = -sum(Y[i] * math.log(y_pred[i] + 1e-9) for i in range(vocab_size))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    dWhy = [[0 for _ in range(vocab_size)] for _ in range(hidden_size)]
    dby = [0 for _ in range(vocab_size)]
    dWxh = [[0 for _ in range(hidden_size)] for _ in range(vocab_size)]
    dWhh = [[0 for _ in range(hidden_size)] for _ in range(hidden_size)]
    dbh = [0 for _ in range(hidden_size)]

    dy = [y_pred[i] - Y[i] for i in range(vocab_size)]
    for i in range(hidden_size):
        for j in range(vocab_size):
            dWhy[i][j] += hs[-1][i] * dy[j]
    for i in range(vocab_size):
        dby[i] += dy[i]

    dh = [sum(Why[i][j] * dy[j] for j in range(vocab_size)) for i in range(hidden_size)]

    for t in reversed(range(len(X))):
        h = hs[t]
        dh_raw = [dh[i] * dtanh(h)[i] for i in range(hidden_size)]
        for i in range(vocab_size):
            for j in range(hidden_size):
                dWxh[i][j] += X[t][i] * dh_raw[j]
        for i in range(hidden_size):
            for j in range(hidden_size):
                prev_h = hs[t-1][j] if t > 0 else 0
                dWhh[i][j] += prev_h * dh_raw[j]
        for i in range(hidden_size):
            dbh[i] += dh_raw[i]
        dh = [sum(Whh[i][j] * dh_raw[j] for j in range(hidden_size)) for i in range(hidden_size)]

    for i in range(vocab_size):
        for j in range(hidden_size):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
    for i in range(hidden_size):
        for j in range(hidden_size):
            Whh[i][j] -= learning_rate * dWhh[i][j]
    for i in range(hidden_size):
        for j in range(vocab_size):
            Why[i][j] -= learning_rate * dWhy[i][j]
    for i in range(hidden_size):
        bh[i] -= learning_rate * dbh[i]
    for i in range(vocab_size):
        by[i] -= learning_rate * dby[i]


print("\nPrediction after training:")
h = [0.0 for _ in range(hidden_size)]
for x_t in X:
    h = tanh(vector_add(matmul(x_t, Wxh), vector_add(matmul(h, Whh), bh)))
y_raw = vector_add(matmul(h, Why), by)
y_pred = softmax(y_raw)
pred_idx = y_pred.index(max(y_pred))
print(f"Predicted word: {idx_to_word[pred_idx]}")


# In[ ]:




