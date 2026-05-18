import numpy as np

number_of_examples_train = 20000
number_of_examples_test = 1000

max_sum = 7
max_base_num = 5

X_train =  np.zeros((number_of_examples_train, (max_base_num+1)*2), dtype=np.uint32)
Y_train = np.zeros(number_of_examples_train, dtype=np.uint32)

X_test =  np.zeros((number_of_examples_test, (max_base_num+1)*2), dtype=np.uint32)
Y_test = np.zeros(number_of_examples_test, dtype=np.uint32)


X_test2 =  np.zeros((number_of_examples_test, (max_base_num+1)*2), dtype=np.uint32)
Y_test2 = np.zeros(number_of_examples_test, dtype=np.uint32)

def reverse_encoding(sample):
    x_sample = sample[:-1]
    y_sample = sample[-1]
    x_len = int(len(x_sample)/2)
    n1 = np.where(x_sample[:x_len]==1)[0][0]
    n2 = np.where(x_sample[x_len:]==1)[0][0]
    print('Sample X :',n1,' + ', n2, ' = ', n1+n2, 'Y:', y_sample)

pairs = []
for base_num in range(max_base_num+1):
    for add_num in range(base_num+1):
        if base_num+add_num < max_sum:
            #print(base_num,'+',add_num,'=',base_num+add_num)
            pairs.append((base_num, add_num))
    #for i in range(number_of_examples):


'''
0 + 0 = 0
1 + 0 = 1
1 + 1 = 2
2 + 0 = 2
2 + 1 = 3
2 + 2 = 4
3 + 0 = 3
3 + 1 = 4 ** taken out
3 + 2 = 5
3 + 3 = 6
4 + 0 = 4
4 + 1 = 5
4 + 2 = 6
5 + 0 = 5
5 + 1 = 6
'''
print(pairs)
train_pairs = pairs.copy()
train_pairs.remove((3,1))
print(len(pairs), len(train_pairs))



for i in range(number_of_examples_train):
    random_index = np.random.choice(len(train_pairs))
    p = train_pairs[random_index]
    X_train[i][p[0]] = 1
    X_train[i][max_base_num+1+p[1]] = 1
    Y_train[i] =p[0]+p[1]

Y_train = np.asarray(Y_train).reshape((number_of_examples_train, 1))

data_train = np.append(X_train, Y_train, axis=1)
        



'''random_index = np.random.choice(number_of_examples_train)
print(random_index)
reverse_encoding(data[random_index])'''

np.savetxt("AdditionTrainingData.txt", data_train, fmt='%d')


for i in range(number_of_examples_test):
    random_index = np.random.choice(len(train_pairs))
    p = pairs[random_index]
    X_test[i][p[0]] = 1
    X_test[i][max_base_num+1+p[1]] = 1
    Y_test[i] =p[0]+p[1]

Y_test = np.asarray(Y_test).reshape((number_of_examples_test, 1))

data_test = np.append(X_test, Y_test, axis=1)
        



'''random_index = np.random.choice(number_of_examples_train)
print(random_index)
reverse_encoding(data[random_index])'''

np.savetxt("AdditionTestingData_trainbased.txt", data_test, fmt='%d')



for i in range(number_of_examples_test):
    random_index = np.random.choice(len(pairs))
    p = pairs[random_index]
    X_test2[i][p[0]] = 1
    X_test2[i][max_base_num+1+p[1]] = 1
    Y_test2[i] =p[0]+p[1]

Y_test2 = np.asarray(Y_test2).reshape((number_of_examples_test, 1))

data_test2 = np.append(X_test2, Y_test2, axis=1)
        



'''random_index = np.random.choice(number_of_examples_train)
print(random_index)
reverse_encoding(data[random_index])'''

np.savetxt("AdditionTestingData_all.txt", data_test2, fmt='%d')



X_test3 =  np.zeros((10, (max_base_num+1)*2), dtype=np.uint32)
Y_test3 = np.zeros(10, dtype=np.uint32)

for i in range(10):
    random_index = np.random.choice(len(pairs))
    p = (3,1)
    X_test3[i][p[0]] = 1
    X_test3[i][max_base_num+1+p[1]] = 1
    Y_test3[i] =p[0]+p[1]

Y_test3 = np.asarray(Y_test3).reshape((10, 1))

data_test3 = np.append(X_test3, Y_test3, axis=1)
        

np.savetxt("AdditionTestingData_limited.txt", data_test3, fmt='%d')
