import numpy as np
import pandas as pd
import itertools


def file_read(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(','))
    # print(len(data))
    data = np.array(data, dtype=float)
    return data  # .reshape([len(data),4,1])

user_size = 15400
user_negative_items = dict(zip([i for i in range(user_size)], [[] for i in range(user_size)]))
user_positive_items = dict(zip([i for i in range(user_size)], [[] for i in range(user_size)]))

user_item_n = file_read('./dev/data_uniform.csv')

for i in range(user_item_n.shape[0]):
    if user_item_n[i][2] == 0:
        user_negative_items[user_item_n[i][0]].append(user_item_n[i][1])
    if user_item_n[i][2] == 1:
        user_positive_items[user_item_n[i][0]].append(user_item_n[i][1])

# print(user_negative_items)
negative_samples = np.zeros([user_item_n.shape[0], 99])
for i in range(user_item_n.shape[0]):
#     print(user_negative_items[user_item_n[i][0]])
#     print(user_negative_items[user_item_n[i][1]])
    if user_negative_items[user_item_n[i][0]] == []:
        a = np.arange(1720)
        a = np.array(list(itertools.compress(a, [i not in user_positive_items[user_item_n[i][0]] for i in range(len(a))])))
        negative_samples[i] = np.random.choice(a, 99, p=None)
    else:
        negative_samples[i] = np.random.choice(user_negative_items[user_item_n[i][0]], 99, p=None)

np.save('./dev/negative_samples.npy', negative_samples)
# a = np.array(list(itertools.compress(a, [i not in index for i in range(len(a))])))