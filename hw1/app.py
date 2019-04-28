# -*- coding: utf-8 -*-
# 20133096 LEE HYUNJAE (이현재)
import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import Counter

# functions


def count_freqency(listinput):
    count_list = list()
    for cnt in listinput:
        # count word freqency
        listinput.count(listinput[cnt][1])
    return count_list


def tupletolist(listinput, i):
    listoutput = list()
    for cnt in listinput:
        listoutput.append(cnt[i])
    return listoutput


def countFreq(listinput):
    w_count = {}
    for lst in listinput:
        try:
            w_count[lst] += 1
        except:
            w_count[lst] = 1
    return w_count


# To get current path.
path = os.getcwd()
filename = input("Write text file nmae (ex:Coco.txt) :")
# Open the file, split words and sort by freqency.
with open(path + "/textfiles/" + filename, 'r', encoding='utf-8') as document_text:
    wordcount = Counter(document_text.read().split())

# Make a list that has tuples (word - count of the word) of pairs
result = sorted(wordcount.items(), key=operator.itemgetter(1), reverse=True)

# Split first value of each tuple and second value of each tuple and save them into diffrent lists.
wordList = tupletolist(result, 0)
freqList = tupletolist(result, 1)

# Count if there are same freqency
freqCnt = countFreq(freqList)

print('word - freqency \n' + str(result) + '\n')
print('key: how many time word(s) appear, value: how many the matched word(s) are\n' + str(freqCnt))

# plotting
pos = np.arange(len(freqCnt.keys()))
width = 2.0
plt.rcParams['axes.grid'] = True
plt.title(filename)
plt.scatter(list(freqCnt.keys()), list(
    freqCnt.values()), s=20, color='b', alpha=0.5)
plt.xlabel('how many times the word appears')
plt.ylabel('freqency of x axis')
plt.show()

document_text.close()
