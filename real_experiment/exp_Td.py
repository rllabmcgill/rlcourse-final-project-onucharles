import scipy.io as sio
import numpy as np
from numba import jit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import copy

STATES = [1, 2, 3, 4, 99]
POS_LABEL = 1
NEG_LABEL = -1

def getStateIdx(stateCode):
    return np.where(STATES == stateCode)

def makeSemiMarkovChain(data):

    n_data = data.shape[0]
    semiMarkovData = copy.deepcopy(data) #np.zeros(data.shape)

    for i in np.arange(n_data):
        print('converting to semimarkov chain...example {0} of {1}'.format(i+1,n_data))
        x_semi = []
        x = data[i, 1]

        prev = x[0]
        x_semi.append(prev)
        for j in np.arange(len(x)):
            cur = x[j]
            if cur != prev:
                x_semi.append(cur)
                prev = cur
        semiMarkovData[i,1] = np.array(x_semi)
        # print(x_semi)
        # print(x)

    #save
    pickle.dump(semiMarkovData, open("semiMarkovData.pkl","wb"))
    return  semiMarkovData

@jit
def tdLearning_batch(trainingData, trainingLabels, alpha, gamma, n_iter):

    state_values = np.zeros(len(STATES))
    #n_iter = 10
    for m in range(n_iter):
        print('learning state values...' + str(m+1) + ' of ' + str(n_iter) + ' iterations')
        for i in range(trainingData.shape[0]):
            sequence = trainingData[i]
            label = trainingLabels[i]
            for j in range(len(sequence)):
                curStateIdx = getStateIdx(sequence[j])

                if j == len(sequence) - 1:
                    reward = label  # get last reward as label
                    state_values[curStateIdx] += alpha * (reward - state_values[curStateIdx])
                    break

                reward = 1e-5
                nextStateIdx = getStateIdx(sequence[j + 1])
                state_values[curStateIdx] += alpha * (reward + gamma * state_values[nextStateIdx] - state_values[curStateIdx])
        print ('state values=' + str(state_values))

    return state_values

@jit
def tdPrediction(data, posMeanReturn, negMeanReturn, state_values):
    #data is with reward at the end.

    seqReturn = calculateCumValue(data, state_values)
    print('test returns' + str(seqReturn))

    predictions = np.zeros(data.shape[0])
    #make prediction by assigning to class that minimises absolute distance from mean.
    for i in range(data.shape[0]):
        posDist = np.abs(seqReturn[i] - posMeanReturn)
        negDist = np.abs(seqReturn[i] - negMeanReturn)
        #predictions[i] = int(posDist < negDist)
        if posDist < negDist:
            predictions[i] = POS_LABEL
        else:
            predictions[i] = NEG_LABEL
        #pred = posDist < negDist
        #label = sequence[-1]
        #correctPredCount += label == pred

    #acc = correctPredCount / len(data)
    return predictions

@jit
def tdPredictionFromTerm(data, posMeanTermValue, negMeanTermValue, state_values):
    #data is with reward at the end.

    seqReturn = calculateTermValue(data, state_values)

    predictions = np.zeros(data.shape[0])
    #make prediction by assigning to class that minimises absolute distance from mean.
    for i in range(data.shape[0]):
        posDist = np.abs(seqReturn[i] - posMeanTermValue)
        negDist = np.abs(seqReturn[i] - negMeanTermValue)

        if posDist < negDist:
            predictions[i] = POS_LABEL
        else:
            predictions[i] = NEG_LABEL
    return predictions

@jit
def calculateCumValue(data, state_values):
    '''
    Do not pass a one-dimensional array to this method.
    :param data:
    :param state_values:
    :return:
    '''
    #data is without reward at the end. just state sequence.

    n_data = data.shape[0]
    cumValue = np.zeros(n_data)
    for i in range(n_data):
        sequence = data[i]
        for j in range(len(sequence)):
            curStateIdx = getStateIdx(sequence[j])
            cumValue[i] += (1/(j+1)) * (state_values[curStateIdx] - cumValue[i])    #running average.

    return cumValue

@jit
def calculateTermValue(data, state_values):
    '''
    Do not pass a one-dimensional array to this method.
    :param data:
    :param state_values:
    :return:
    '''

    n_data = data.shape[0]
    termValue = np.zeros(n_data)
    for i in range(n_data):
        sequence = data[i]
        termStateIdx = getStateIdx(sequence[-1])
        termValue[i] = state_values[termStateIdx]

    return termValue

def getDataLabels(data):
    '''
    Retrieve labels from data. Change 0 to -1
    :param data:
    :return:
    '''
    labels = np.zeros(len(data))
    for i in np.arange(len(data)):
        y = data[i,2][0]
        if y == 0:
            labels[i] = NEG_LABEL
        elif y == 1:
            labels[i] = POS_LABEL

    return labels


# def run_exp(data):
#
#     y = getDataLabels(data)
#     X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.33, random_state = 42)
#
#     allIndices = np.arange(len(data))
#     state_values = np.zeros(5)
#     n_iter = 10
#     for m in range(n_iter):
#         print('learning state values...' + str(m+1) + ' of ' + str(n_iter) + ' iterations')
#         for i in np.arange(len(data)):
#             trainingData = data[allIndices != i]
#             testData = data[allIndices == i]
#
#             learned_values = tdLearning_batch(trainingData, state_values, 0.05)
#             posTrainData=0
#             negTrainData=0
#             posDataCumValue = calculateReturn(posTrainData, learned_values)
#             negDataCumValue = calculateReturn(negTrainData, learned_values)
#
#             acc = tdPrediction(testData, np.mean(posDataCumValue), np.mean(negDataCumValue),
#                                learned_values)

def run_exp(data):

    y = getDataLabels(data)
    x = data[:, 1]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    # k = 5
    # skf = StratifiedKFold(n_splits=k)

    n_iter = 10
    alpha = 0.01
    gamma = 1

    y_pred = np.zeros(len(y))

    # i = 1
    # for trainIdx, testIdx in skf.split(x, y):
    # print('fold {0} of {1}'.format(i, k))
    # X_train, X_test = x[trainIdx], x[testIdx]
    # y_train, y_test = y[trainIdx], y[testIdx]
    all_idx = np.arange(len(y))
    for i in all_idx:
        print('leave one out cross validation...{0} of {1} tests'.format(i+1, len(y)))
        X_train, X_test = x[all_idx != i], x[all_idx == i]
        y_train, y_test = y[all_idx != i], y[all_idx == i]

        state_values = tdLearning_batch(X_train, y_train, alpha, gamma, n_iter)
        posTrainData = X_train[y_train == POS_LABEL]
        negTrainData = X_train[y_train == NEG_LABEL]
        posDataCumValue = calculateCumValue(posTrainData, state_values)
        negDataCumValue = calculateCumValue(negTrainData, state_values)
        centerPosCumValues = np.median(posDataCumValue)
        centerNegCumValues = np.median(negDataCumValue)

        print('median of pos cum values: ' + str(centerPosCumValues))
        print('median of neg cum values: ' + str(centerNegCumValues))
        print('mean of pos cum values: ' + str(np.mean(posDataCumValue)))
        print('mean of neg cum values: ' + str(np.mean(negDataCumValue)))
        predictions = tdPrediction(X_test, centerPosCumValues, centerNegCumValues, state_values)

        #display preditions from current folds (relevant only if not leave-one-out)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[NEG_LABEL, POS_LABEL]).ravel()
        print('tn={0}; fp={1}; fn={2}; tp={3}'.format(tn, fp, fn, tp))

        #keep predictions from all folds.
        y_pred[i] = predictions
        #i+=1

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[NEG_LABEL,POS_LABEL]).ravel()    #labels = [neg, pos]
    sensitivity = tp /(tp + fn)
    specificity = tn /(tn + fp)
    acc = (tp + tn)/(tp + tn + fp + fn)

    print('results for gamma = ' + str(gamma))
    print('tn={0}; fp={1}; fn={2}; tp={3}'.format(tn, fp, fn, tp))
    print('sensitivity', str(sensitivity) + '; specificity=' + str(specificity) + 'accuracy=' + str(acc))
    pickle.dump((tn, fp, fn, tp), open("result_gamma_" + str(gamma) + ".pkl", "wb"))
    #favorite_color = pickle.load(open("save.p", "rb"))

    #plot histogram of cumulative values
    plt.hist(posDataCumValue, bins=10, label='positive')  # plt.hist passes it's arguments to np.histogram
    plt.hist(negDataCumValue, bins=10, label='negative')  # plt.hist passes it's arguments to np.histogram
    plt.legend(loc=2)
    plt.show()

def run_exp_best_gamma(data):

    y = getDataLabels(data)
    x = data[:, 1]
    k = 5
    skf = StratifiedKFold(n_splits=k)

    n_iter = 20
    alpha = 0.01
    gamma = 1

    y_pred = np.zeros(len(y))

    i = 1
    for trainIdx, testIdx in skf.split(x, y):
        print('fold {0} of {1}'.format(i, k))
        X_train, X_test = x[trainIdx], x[testIdx]
        y_train, y_test = y[trainIdx], y[testIdx]

        state_values = tdLearning_batch(X_train, y_train, alpha, gamma, n_iter)
        posTrainData = X_train[y_train == POS_LABEL]
        negTrainData = X_train[y_train == NEG_LABEL]
        posDataCumValue = calculateCumValue(posTrainData, state_values)
        negDataCumValue = calculateCumValue(negTrainData, state_values)
        centerPosCumValues = np.median(posDataCumValue)
        centerNegCumValues = np.median(negDataCumValue)

        print('median of pos cum values: ' + str(centerPosCumValues))
        print('median of neg cum values: ' + str(centerNegCumValues))
        print('mean of pos cum values: ' + str(np.mean(posDataCumValue)))
        print('mean of neg cum values: ' + str(np.mean(negDataCumValue)))
        predictions = tdPrediction(X_test, centerPosCumValues, centerNegCumValues, state_values)

        #display preditions from current folds (relevant only if not leave-one-out)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[NEG_LABEL, POS_LABEL]).ravel()
        print('tn={0}; fp={1}; fn={2}; tp={3}'.format(tn, fp, fn, tp))

        #keep predictions from all folds.
        y_pred[testIdx] = predictions
        i+=1
        break

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[NEG_LABEL,POS_LABEL]).ravel()    #labels = [neg, pos]
    sensitivity = tp /(tp + fn)
    specificity = tn /(tn + fp)
    acc = (tp + tn)/(tp + tn + fp + fn)

    print('kfold results for gamma = ' + str(gamma))
    print('tn={0}; fp={1}; fn={2}; tp={3}'.format(tn, fp, fn, tp))
    print('sensitivity', str(sensitivity) + '; specificity=' + str(specificity) + 'accuracy=' + str(acc))
    pickle.dump((tn, fp, fn, tp), open("result_gamma_" + str(gamma) + ".pkl", "wb"))
    #favorite_color = pickle.load(open("save.p", "rb"))

    #plot histogram of cumulative values
    plt.hist(posDataCumValue, bins=10, label='positive')  # plt.hist passes it's arguments to np.histogram
    plt.hist(negDataCumValue, bins=10, label='negative')  # plt.hist passes it's arguments to np.histogram
    plt.legend(loc=2)
    plt.show()


def plotCvResults():
    sensitivities = []
    specificities = []
    x_data = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.plot(x_data, sensitivities, '-o', label='Sensitivity')
    plt.plot(x_data, specificities, '-o', label='Specificity')
    plt.ylabel('performance')
    plt.xlabel('discount factor (\gamma)')
    plt.legend(loc='upper left')
    plt.show()

data = sio.loadmat('../../Data/automated_states_sbt_7days_criteria.mat')
data = data['data'] #array of arrays.
#run_exp(data)
run_exp_best_gamma(data)

#semi_data = makeSemiMarkovChain(data)

# semi_data = pickle.load(open('semiMarkovData.pkl', 'rb'))
# run_exp(semi_data)

#Next:
#2. use same value to learn for semi-Markov chain. (use mean not median for this).
#3. try td lambda should be straightfoward (need to cross validate lambda)