import numpy as np
from toy_environment import toy_environment
import matplotlib.pyplot as plt
import random

def fitTransitionModel(examples, n_states):
    transition_sum = np.zeros((n_states, n_states))

    for i in np.arange(len(examples)):
        sequence = examples[i]

        prevState = sequence[0]
        for j in range(1,len(sequence)):
            curState = sequence[j]

            transition_sum[prevState, curState] += 1
            prevState = curState

    sum_over_next_states = np.sum(transition_sum, 1)
    sum_over_next_states = sum_over_next_states.reshape((n_states, 1))  #reshape to enable next (divide) step.
    transition_prob = transition_sum / sum_over_next_states
    #transition_prob[transition_prob == np.nan] = 0  # set to zero all divide by zeros (nans)
    #print(transition_prob)
    return transition_prob#, transition_sum

def computeLikelihood(examples, transition_prob):
    n_examples = len(examples)
    likelihoods = np.zeros((n_examples))

    for i in np.arange(len(examples)):
        sequence = examples[i]

        likelihood = 0
        prevState = sequence[0]
        for j in range(1,len(sequence)):
            curState = sequence[j]

            if transition_prob[prevState, curState] == 0:
                continue
            likelihood += np.log10(transition_prob[prevState, curState])
            prevState = curState

        likelihoods[i] = likelihood
    return likelihoods

def makeDataForTd(posData, negData):
    posDataR = []
    negDataR = []
    # add labels as last item in list.
    for i in range(len(posData)):
        seq = posData[i]
        posDataR.append(seq + [1])
    for i in range(len(negData)):
        seq = negData[i]
        negDataR.append(seq + [0])

    # merge and shuffle them
    data = negDataR + posDataR
    random.shuffle(data)
    return data

def tdLearning(trainingData, alpha, tenv):
    avg_state_values = np.zeros(tenv.n_states)
    # state_values = np.zeros(tenv.n_states)
    for k in range(1):
        state_values = np.zeros(tenv.n_states)
        for i in range(len(trainingData)):
            sequence = trainingData[i]
            for j in range(len(sequence)):
                curState = sequence[j]

                if j == tenv.seq_length - 1:
                    reward = sequence[j + 1]  # because reward is stored as very last item in each sequence.
                    state_values[curState] += alpha * (reward - state_values[curState])
                    break
                reward = 0
                nextState = sequence[j + 1]
                state_values[curState] += alpha * (reward + 1 * state_values[nextState] - state_values[curState])

        # running mean of state values.
        avg_state_values += (1 / (k + 1)) * (state_values - avg_state_values)
    return avg_state_values

def tdLearning_batch(trainingData, alpha, tenv):
    avg_state_values = np.zeros(tenv.n_states)
    # state_values = np.zeros(tenv.n_states)
    for k in range(1):
        state_values = np.zeros(tenv.n_states)
        for m in range(1000):
            for i in range(len(trainingData)):
                sequence = trainingData[i]
                for j in range(len(sequence)):
                    curState = sequence[j]

                    if j == tenv.seq_length - 1:
                        reward = sequence[j + 1]  # because reward is stored as very last item in each sequence.
                        state_values[curState] += alpha * (reward - state_values[curState])
                        break
                    reward = 0
                    nextState = sequence[j + 1]
                    state_values[curState] += alpha * (reward + 1 * state_values[nextState] - state_values[curState])

        # running mean of state values.
        avg_state_values += (1 / (k + 1)) * (state_values - avg_state_values)
    return avg_state_values

def tdPrediction(data, posMeanReturn, negMeanReturn, state_values):
    #data is with reward at the end.

    correctPredCount = 0
    #make prediction by assigning to class that minimises absolute distance from mean.
    for i in range(len(data)):
        sequence = data[i]
        seqReturn = calculateReturn([sequence[:-1]], state_values)
        posDist = np.abs(seqReturn - posMeanReturn)
        negDist = np.abs(seqReturn - negMeanReturn)
        pred = posDist < negDist
        label = sequence[-1]
        correctPredCount += label == pred

    acc = correctPredCount / len(data)
    return acc

# def tdPredictionWeightedReturn(data, posMeanReturn, negMeanReturn, state_values, posWeights, negWeights):
#     #data is with reward at the end.
#
#     correctPredCount = 0
#     #make prediction by assigning to class that minimises absolute distance from mean.
#     for i in range(len(data)):
#         sequence = data[i]
#         posReturn, negReturn = calculateWeightedReturn([sequence[:-1]], state_values, posWeights, negWeights)
#         posDist = np.abs(posReturn - posMeanReturn)
#         negDist = np.abs(negReturn - negMeanReturn)
#         pred = posDist < negDist
#         label = sequence[-1]
#         correctPredCount += label == pred
#
#     acc = correctPredCount / len(data)
#     return acc

def calculateReturn(data, state_values):
    #data is without reward at the end. just state sequence.

    n_data = len(data)
    cumRewards = np.zeros(n_data)
    for i in range(n_data):
        sequence = data[i]
        for j in range(len(sequence)):
            curState = sequence[j]
            cumRewards[i] += state_values[curState]

    return cumRewards

# def calculateWeightedReturn(data, state_values, pos_weights, neg_weights):
#     #data is without reward at the end. just state sequence.
#
#     n_data = len(data)
#     posReturn = np.zeros(n_data)
#     negReturn = np.zeros(n_data)
#     for i in range(n_data):
#         sequence = data[i]
#         for j in range(len(sequence)):
#             curState = sequence[j]
#             posReturn[i] += pos_weights[curState] * state_values[curState]
#             negReturn[i] += neg_weights[curState] * state_values[curState]
#
#     return posReturn, negReturn

def runTdExp():
    seq_len = 5
    tenv = toy_environment(seq_len)

    n_runs = 1 #500
    training_sizes = [100]
    #training_sizes = np.linspace(1, 100, 10, dtype=int)
    accs = np.zeros(len(training_sizes))

    #generate fixed test set.
    n_test_examples = 50
    posTestData, negTestData = tenv.generateExamples(n_test_examples)
    testData = makeDataForTd(posTestData, negTestData)

    for j in range(n_runs):
        print('run ' + str(j) + '...')
        for i in range(len(training_sizes)):
            n_train_examples = training_sizes[i]
            posTrainData, negTrainData = tenv.generateExamples(n_train_examples)

            #add final reward and shuffle data
            trainingData = makeDataForTd(posTrainData, negTrainData)

            # pass to td algorithm to learn weights
            alpha = 0.01
            #avg_state_values = tdLearning(trainingData, alpha, tenv)
            avg_state_values = tdLearning_batch(trainingData, alpha, tenv)
            print(avg_state_values)
            #avg_state_values = [0.12267084, 0.12169358, 0.18033124]

            posDataCumRewards = calculateReturn(posTrainData, avg_state_values)
            negDataCumRewards = calculateReturn(negTrainData, avg_state_values)
            # print(posDataCumRewards)
            # print(negDataCumRewards)
            # print(np.mean(posDataCumRewards))
            # print(np.mean(negDataCumRewards))

            plt.hist(posDataCumRewards, bins=10, label='positive')  # plt.hist passes it's arguments to np.histogram
            plt.hist(negDataCumRewards, bins=10, label='negative')  # plt.hist passes it's arguments to np.histogram
            plt.legend(loc=2)
            plt.show()
            acc = tdPrediction(testData, np.mean(posDataCumRewards), np.mean(negDataCumRewards),
                               avg_state_values)
            accs[i] += (1/(j+1)) * (acc - accs[i])
        print(accs)

    plt.plot(np.arange(len(training_sizes)), accs)
    plt.xticks(np.arange(len(training_sizes)), training_sizes)
    plt.show()

def runMarkovExp():
    tenv = toy_environment(5)
    n_runs = 100
    training_sizes = np.linspace(1, 100, 10, dtype=int)
    avg_acc_per_size = np.zeros((len(training_sizes)))

    for j in np.arange(len(training_sizes)):
        # print('Training size: ' + training_sizes[j])
        accs = np.zeros((n_runs))
        for i in np.arange(n_runs):
            # generate training examples and fit models
            posTrainData, negTrainData = tenv.generateExamples(training_sizes[j])
            pos_transition_prob = fitTransitionModel(posTrainData, 3)
            neg_transition_prob = fitTransitionModel(negTrainData, 3)

            # generate test examples and predict
            posTestData, negTestData = tenv.generateExamples(50)

            # prediction on positive test data
            pos_likelihoods = computeLikelihood(posTestData, pos_transition_prob)
            neg_likelihoods = computeLikelihood(posTestData, neg_transition_prob)
            predictions = pos_likelihoods > neg_likelihoods
            labels = np.ones((len(posTestData)))
            acc1 = sum(predictions == labels) / len(posTestData)

            # prediction on negative test data
            pos_likelihoods = computeLikelihood(negTestData, pos_transition_prob)
            neg_likelihoods = computeLikelihood(negTestData, neg_transition_prob)
            predictions = pos_likelihoods > neg_likelihoods
            labels = np.zeros((len(negTestData)))
            acc2 = sum(predictions == labels) / len(negTestData)
            acc = (acc1 + acc2) / 2.0
            accs[i] = acc
        avg_acc_per_size[j] = np.mean(accs)
    print(avg_acc_per_size)

    plt.plot(training_sizes, avg_acc_per_size, '-o')
    plt.ylabel('accuracy')
    plt.xlabel('training set size')
    #plt.legend(loc='upper left')
    plt.show()

#runMarkovExp()
runTdExp()


#result shown in report
# training_sizes = np.linspace(1, 100, 10)
# avg_acc_per_size = [ 0.6108 ,  0.69548,  0.71548,  0.72936,  0.72708,  0.73428 , 0.73688,  0.74628,
#   0.74436,  0.75272]
# plt.plot(training_sizes, avg_acc_per_size, '-o', label='online TD(0)')
# avg_acc_per_size = [ 0.63541667,  0.77416667 , 0.79520833 , 0.80520833 , 0.81916667,  0.821875,
#   0.83729167,  0.833125,    0.84708333 , 0.85458333]
# plt.plot(training_sizes, avg_acc_per_size, '-o', label='batch TD(0)')
# avg_acc_per_size = [ 0.4542 , 0.6734,  0.82,    0.8426,  0.8578,  0.8614,  0.868 ,  0.862,   0.8606,
#   0.8604]
# plt.plot(training_sizes, avg_acc_per_size, '-o', label='markov chain model')

# plt.ylabel('accuracy')
# plt.xlabel('training set size')
# plt.legend(loc='lower right')
#plt.ylim(0.4,0.9)
# plt.show()
#
# plt.hist(posDataCumRewards, bins=10, label='positive')  # plt.hist passes it's arguments to np.histogram
# plt.hist(negDataCumRewards, bins=10, label='negative')  # plt.hist passes it's arguments to np.histogram
# plt.legend(loc=2)
# plt.show()