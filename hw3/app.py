# -*- coding: utf-8 -*-
# 20133096 이현재
'''
With the same data collection (obtained in HW1), please write a code for ranking the documents with "Vector Space model".

In your report, you have to explain how to design your vector space with N dimension.
For stopword removal, you can use [1].
Test your rankings with your own queries.
Pros? Why?
Cons? Why?
What have you tried to improve your ranking?

3. You should NOT use any external libraries (e.g., Lucene).
'''

import os  # built-in
import math  # built-in
import operator  # built-in
import numpy as np  # allowed to use

path = os.getcwd()

filenames = ["Coco.txt", "Big Sick.txt", "Beauty and the Beast.txt", "Get Out.txt", "Guardians of the Galaxy Vol 2.txt",
             "It.txt", "La La Land.txt", "Logan.txt", "Thor Ragnarok.txt", "War for the Planet of the Apes.txt"]

vector_space_model = []
idf_space_model = []
tf_space_model = []
df_result = []
word_result = []


# stopwords
file_stopwords = open("stopwords.txt", 'r')
data = file_stopwords.read()
stopwords_list = data.split(',')
file_stopwords.close()


def load(filename, query, iteration):
    document_text = open(path + "/textfiles/" +
                         filename, 'r', encoding='utf-8-sig')
    wordcount = {}
    words = []
    freq = []
    idf_result = []
    tf_result = []

    data = document_text.read()
    document_text.close()

    listofwords = data.split()
    listofwords = preprocessing_delete(listofwords)
    # print(listofwords)

    for word in listofwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

    # Append words and correspending freqency of them to diffrent array
    for k, v in wordcount.items():
        # Vector space model of this document
        words.append(k)
        freq.append(v)

    # Calculate TF, IDF
    tf = 0
    df = 0

    for q in query:
        for word in words:
            df = 0
            tf = 0
            if word == q:
                df = 1  # in case this doc has this specific query
                tf += freq[words.index(word)]

                # idf = log2(N/df)
                idf_result.append(df)
                break
            else:
                # if the selected word is the last word in words
                if(words.index(word) == len(words)-1):
                    idf_result.append(0)
                    tf = 0
                else:
                    continue
        tf_result.append(tf)

    for i in range(len(query)):
        idf_space_model[i][iteration] = idf_result[i]
        tf_space_model[i][iteration] = tf_result[i]
    # Input them into lists
    word_result.append(words)
    df_result.append(tf_result)
    # vector_space_model.append()


def preprocessing_delete(results):
    # Delete stopwords : DONE
    output = []

    for check in results:
        check = check.replace("'", "").replace(",", "").replace(".", "").replace("(", "").replace(
            ")", "").replace("-", "").replace("?", "").replace("!", "").replace(":", "").replace("/", "").replace("\"", "").replace(" ", "").lower()
        output.append(check)

    for stopword in stopwords_list:
        for check in output:
            if(check == stopword):
                del output[output.index(check)]
            elif(check == ""):
                del output[output.index(check)]
    # print(output)
    return output


#query_raw = input(" Write your query :")
query_raw = open("query.txt", 'r', encoding='utf-8-sig').read()
query_before_pre = query_raw.split()
query = preprocessing_delete(query_before_pre)
print(" Your query is considered as " + str(query))


def excute():
    for i in range(10):
        load(filenames[i], query, i)


def assigne_model(listinput, row, col):
    c = [[0 for i in range(col)] for j in range(row)]

    return c


def idf_calculate(idf_space_model):
    output = []

    for element in idf_space_model:
        try:
            output.append(math.log(10/sum(element), 2))
        except ZeroDivisionError:
            output.append(0)

    return output


def tf_calculate(tf_space_model):
    output = []

    for element in tf_space_model:
        temp = []
        for i in range(10):
            if(element[i] != 0):
                element[i] = 1 + math.log(element[i])
                temp.append(element[i])
            else:
                temp.append(0)
        output.append(temp)
    return output


def vector_space_model_calculate(
        vector_space_model, idf_model, tf_model):

    weight = []

    for i in range(len(idf_model)):
        value = idf_model[i]
        list_temp = tf_model[i]
        temp = []
        for k in range(10):  # number of documents
            input_value = list_temp[k] * value
            temp.append(input_value)
        weight.append(temp)
    return weight


def df_calculate(df_result, length):
    output = []

    for i in range(length):
        value = 0
        valuetemp = 0
        for n in range(10):  # number of documents
            value = df_result[n][i]
            valuetemp += value
        output.append(valuetemp)

    return output


def query_length(idf_model, df_freq, query):

    w_count = {}
    lenght_Query = []
    temp = []
    output = 0

    # count query
    for lst in query:
        try:
            w_count[lst] += 1
        except:
            w_count[lst] = 1

    # calculate length_query
    for key, value in w_count.items():
        #print(key, value)
        # number of the query(value) / number of query in all docs * idf
        index = query.index(key)
        try:
            lenght = value / df_freq[index] * idf_model[index]
        except ZeroDivisionError:
            lenght = 0
        temp.append(lenght)

    for item in temp:
        output += pow(item, 2)
    output = math.sqrt(output)

    # additional work for cosSim
    for i in range(len(query)):
        try:
            lenght = value / df_freq[i] * idf_model[i]
        except ZeroDivisionError:
            lenght = 0
        lenght_Query.append(lenght)

    return output, lenght_Query


def doc_length(vector_space_model, length):
    output = []

    for i in range(10):
        value = 0
        valuetemp = 0
        for j in range(length):
            valuetemp = vector_space_model[j][i]
            value += pow(valuetemp, 2)
        output.append(math.sqrt(value))

    return output


def dot_product(v1, v2):
    return sum(map(operator.mul, v1, v2))


def vector_cos(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))

    return prod / (len1 * len2)


# Assign vector-space-model and idf-space-model
vector_space_model = assigne_model(vector_space_model, len(query), 10)
idf_space_model = assigne_model(idf_space_model, len(query), 10)
tf_space_model = assigne_model(tf_space_model, len(query), 10)

# pre-processing
excute()

idf_model = idf_calculate(idf_space_model)  # done
tf_model = tf_calculate(tf_space_model)  # done
df_freq = df_calculate(df_result, len(query))  # done

vector_space_model = np.array(vector_space_model_calculate(
    vector_space_model, idf_model, tf_model))  # done

length_query, query_vector = query_length(idf_model, df_freq, query)
length_doc = doc_length(vector_space_model, len(query))

# # Round lists
length_query = round(length_query, 3)
query_vector = ['%.3f' % elem for elem in query_vector]
length_doc = ['%.3f' % elem for elem in length_doc]

#cosSim(vector_space_model,lenght_query,length_doc ,query_vector)

print("length query ---")
print(length_query)

print("query_vector ---")
print(query_vector)

print("length_doc ---")
print(length_doc)

# set numpy printing options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Singular Value Decomposition
# the reduced or trucated SVD operation can save time by ignoring all the
# extremly small or exactly zero values
print("--- REDUCED SVD ---")
U, s, VT = np.linalg.svd(vector_space_model, full_matrices=False)

print("U:\n {}".format(U))
print("s:\n {}".format(s))
print("VT:\n {}".format(VT))

print("----LSA-----")
K = int(input("K is "))  # K << N || N is # query words

while K > len(query_vector):
    print("K should be smaller then N")
    K = int(input())

print("VT:\n {}".format(np.transpose(VT)))

# Rows of V holds eigenvector values. These are the coordinates of individual document vectors, hence
document_vectors = []

for k in range(len(length_doc)):
    document_vectors.append(list(np.transpose(VT)[k][:K]))
    document_vectors[k] = np.array(document_vectors[k])
    document_vectors[k] = ['%.3f' % elem for elem in document_vectors[k]]

print("Document_Vectors : \n {}".format(document_vectors))
print("Query_Vectors : ", query_vector[:K])

# Cosine similarity
cos = []
v2 = [float(i) for i in query_vector]
for k in range(len(length_doc)):
    v1 = [float(i) for i in document_vectors[k]]
    cos.append(round(vector_cos(v1, v2), 5))

# PRINT RESULT
seq = sorted(cos, reverse=True)
print("Cos similarity by LSI for 10 documents with given query :: ")
for i in range(len(cos)):
    print(i, " TH  == Doc#", cos.index(seq[i]), " with value ", seq[i])
