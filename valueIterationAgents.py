# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        # I start off from the bottom and keep looping until I reach the desired iteration
        for Iteration_ in range(0, self.iterations):
            self.copyValues = self.values.copy()  # Make a copy of the util.Counter()
            States = self.mdp.getStates()  # This is the (s) portion of the algorithm

            # I loop through each state (s) and get the possible actions which forms a Q state (s,a)
            for State_ in States:
                actionList = self.mdp.getPossibleActions(State_)

                if self.mdp.isTerminal(State_) is False:  # Terminal States return 0
                    qValueList = []

                    for Action_ in actionList:  # I loop through the Q-states of each state

                        # I am getting (s') and T(s,a,s') and I initialize the Q value

                        nextStatesandTransitions = self.mdp.getTransitionStatesAndProbs(State_, Action_)
                        qValue = 0

                        # I simply apply the formula to calculate the Q-value and I used the Reward function to get R(s,a,s')

                        for (nextAction, Probabilty_) in nextStatesandTransitions:
                            qValue += Probabilty_ * (self.mdp.getReward(State_, Action_, nextAction) + self.discount *self.copyValues[nextAction])

                        # The V(s) start is the maximum of the chance nodes shown in line 83
                        qValueList.append(qValue)

                    # I update the V(s) value in line 85 with the maximum chance node :)
                    self.values[State_] = max(qValueList)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # This is the same as lines 76-86 except without the .copy() portion, this is used in the autograder

        nextStatesandTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0

        for (nextAction, Probabilty_) in nextStatesandTransitions:
            qValue += Probabilty_ * (self.mdp.getReward(state, action, nextAction) + self.discount * self.values[nextAction])
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # This is the same as lines 70-92, but in this case I return the Action corresponding to the best Q-Value as opposed to just wanting the best value as in
        # line 85

        actionList = self.mdp.getPossibleActions(state)

        if self.mdp.isTerminal(state):
            return 0
        else:
            qValueList = []
            for Action_ in actionList:
                qValue = self.computeQValueFromValues(state, Action_)
                qValueList.append(qValue)
            indexAction = qValueList.index(max(qValueList))
            return (actionList[indexAction])

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Same this as ValueIterationAgent except in each iteration I am only adjusting one state
        # I am not doing a batch update as I did previously

        Counter = 0 # I initialize this Counter to cycle through a state per iteration

        for Iteration_ in range(0, self.iterations):
            self.copyValues = self.values.copy()
            States = self.mdp.getStates()
            State_ = States[Counter]
            actionList = self.mdp.getPossibleActions(States[Counter])

            if self.mdp.isTerminal(State_) is False:
                qValueList = []
                for Action_ in actionList:
                    nextStatesandTransitions = self.mdp.getTransitionStatesAndProbs(State_, Action_)
                    qValue = 0
                    for (nextAction, Probabilty_) in nextStatesandTransitions:
                        qValue += Probabilty_ * (self.mdp.getReward(State_, Action_, nextAction) + self.discount * self.copyValues[nextAction])
                    qValueList.append(qValue)
                self.values[State_] = max(qValueList)

            # Increment to the next state
            Counter += 1

            if Counter == len(States): # if I exceeded the indices in the list, start from the beginning again
                Counter = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):



        predecessors_Dict = {state: [] for state in self.mdp.getStates()}

        # Here I calculate all the predecessors for each non terminal state #
        for State_ in self.mdp.getStates():

            if self.mdp.isTerminal(State_) is False:

                for Action_ in self.mdp.getPossibleActions(State_):

                    for (nextState_, Probability) in self.mdp.getTransitionStatesAndProbs(State_, Action_):

                        # Can only be a predecessor if the Probability is greater than 0 and it doesnt already exist in the dictionary
                        if Probability > 0 and nextState_ in predecessors_Dict.keys():

                            # The State_ is considered a predecessor of nextState_ because you can get from State_ -> nextState_ via some
                            # action!
                            if State_ not in predecessors_Dict[nextState_]:
                                predecessors_Dict[nextState_].append(State_)

        priorityQueue = util.PriorityQueue()

        # We prioritize updating states with a higher error in this block and itialize the Priority Queue

        for State_ in self.mdp.getStates():
            if self.mdp.isTerminal(State_) is False:
                qValueList = []

                for Action_ in self.mdp.getPossibleActions(State_):
                    qValueList.append(self.computeQValueFromValues(State_, Action_))

                diff = abs(self.values[State_] - max(qValueList))
                priorityQueue.update(State_, -1 * diff)

        for Iteration in range(0, self.iterations):
            if priorityQueue.isEmpty():
                break

            popped_State = priorityQueue.pop()
            qValueList = []

            for Action_ in self.mdp.getPossibleActions(popped_State):
                qValueList.append(self.computeQValueFromValues(popped_State, Action_))
            self.values[popped_State] = max(qValueList)

            for predecessor in predecessors_Dict[popped_State]:

                if self.mdp.isTerminal(predecessor) is False:
                    poppedQValueList = []

                    for Action_Pred in self.mdp.getPossibleActions(predecessor):
                        poppedQValueList.append(self.computeQValueFromValues(predecessor, Action_Pred))

                    diff = abs(self.values[predecessor] - max(poppedQValueList))

                    # We only want to push if diff > theta, as stated in the algorithm
                    if diff > self.theta:
                        priorityQueue.update(predecessor, -1 * diff)

