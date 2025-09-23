import pickle

with open('../dataset/wd/networks', 'rb') as f:
    data = pickle.load(f)

print(type(data))      # tuple
print(len(data))       # 看看有几个元素
print(type(data[0]))   # 第一个元素是否是 networkx 图


with open('../dataset/wd/attrs', 'rb') as f:
    dict1 = pickle.load(f)

print(type(dict1))   # dict
print(len(dict1))   # 看看某个具体的值
for i in range(10000,10005):
    print(dict1[i])

with open('../dataset/wd/stop_words_cn.pkl', 'rb') as f:
    list1 = pickle.load(f)

print(type(list1))   # dict
print(len(list1))   # 看看某个具体的值
for i in range(0, 200):
    print(list1[i])
