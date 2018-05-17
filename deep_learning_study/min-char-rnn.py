"""
Minimal character-level demo. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
# should be simple plain text file
data = open(
    r'deep_learning_study/datasource/data.txt', 'r', encoding='UTF-8').read()
chars = list(set(data))
print('%d unique characters in data.' % (len(chars), ))
# 不重复字符的长度
vocab_size = len(chars)
# 文本长度
data_size = len(data)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 50  # size of hidden layer of neurons
seq_length = 20  # number of steps to unroll the RNN for
base_learning_rate = 0.01
# every 1000 iteration learning rate gets divided by this
# 每1000次迭代学习率除以这个
learning_rate_decay = 0.85

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    # 前向不断循环，计算loss
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])  # softmax ("cross-entropy loss")
    # backward pass: compute gradients going backwards
    # 后向计算参数的变化
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(
        Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    从模型中提取一个整数序列
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
while n < 20000:
    # prepare inputs (we're sweeping from left to right in steps seq_length
    # long)
    if p + seq_length + 1 >= data_size or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # 当前字符的下一个真实的字符
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # sample from the model now and then
    # 现在从模型中取样
    # 每训练100次，用一个输入去预测一个40个字符长度的串
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 40)
        print('sample:')
        print(''.join(ix_to_char[ix] for ix in sample_ix))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    if p == 0:
        print('iter %d, loss: %f' % (n, loss))  # print progress each epoch

    # perform parameter update with vanilla SGD, decay learning rate
    learning_rate = base_learning_rate * np.power(learning_rate_decay,
                                                  n / 1000.0)
    # 根据学习率优化Wxh，Whh，bh，by等的参数，每一个seq_length，更新一次学习率
    for param, dparam in zip([Wxh, Whh, Why, bh, by],
                             [dWxh, dWhh, dWhy, dbh, dby]):
        param += -learning_rate * dparam

    p += seq_length  # move data pointer
    n += 1  # iteration counter

# 用该模型预测一个100个字符长度的串，其中的0是数组中第0个字符，用该字符预测一个串
print('self_sample:')
print(''.join(
    ix_to_char[i] for i in sample(np.zeros((hidden_size, 1)), 0, 100)))
