import numpy as np

pos_transition_prob = np.array([[0, 0.2, 0.8], [0.2, 0, 0.8], [0.5, 0.5, 0]])
neg_transition_prob = np.array([[0, 0.8, 0.2], [0.8, 0, 0.2], [0.5, 0.5, 0]])

class toy_environment(object):

    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.n_states = 3

    def generateSequence(self, seq_class):
        transition_prob = np.zeros((3,3))

        if seq_class == 1:
            transition_prob = pos_transition_prob
        elif seq_class == 0:
            transition_prob = neg_transition_prob

        sequence = []
        initialState = int(np.random.choice([0, 1, 2], 1))
        curState = initialState
        sequence.append(curState)
        for i in range(1,self.seq_length):
            nextStateProbs = transition_prob[curState, :]
            nextState = int(np.random.choice([0, 1, 2], 1, p=nextStateProbs))
            sequence.append(nextState)
            curState = nextState
        return sequence

    def generateExamples(self, n):
        half_size = np.ceil(n/2)

        pos_examples = []
        for i in np.arange(half_size):
            pos_examples.append((self.generateSequence(1)))

        neg_examples = []
        for i in np.arange(half_size):
            neg_examples.append((self.generateSequence(0)))

        return pos_examples, neg_examples

#print(generateExamples(10,15))
