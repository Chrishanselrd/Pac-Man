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


import mdp
import util

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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # Create a new counter to store updated values
            newValue = util.Counter()

            # Iterate over all states in the MDP
            for state in self.mdp.getStates():
                # If the state is terminal, set the value to 0
                # Otherwise, calculate the maximum expected future reward
                if self.mdp.isTerminal(state):
                    newValue[state] = 0
                else:
                    # Initialize the maximum value as negative infinity
                    maxValue = -float('inf')

                    # Iterate over all possible actions for the current state
                    for action in self.mdp.getPossibleActions(state):
                        # Get the transition states and probabilities for the current state-action pair
                        transitions = self.mdp.getTransitionStatesAndProbs(
                            state, action)
                        value = 0

                        # Calculate the value for the next state by discounting the future rewards
                        for next_state, probability in transitions:
                            reward = self.mdp.getReward(
                                state, action, next_state)
                            value += probability * \
                                (reward + self.discount *
                                 self.values[next_state])

                        # Update the maximum value with the highest value found among all possible actions
                        maxValue = max(maxValue, value)

                    # Set the value of the current state to the maximum value found
                    newValue[state] = maxValue

            # Update the values with the new values calculated in this iteration
            self.values = newValue

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
        "*** YOUR CODE HERE ***"
        qValue = 0
        # Iterate over the possible next states and their corresponding probabilities
        # Calculate the Q-value as the weighted sum of immediate reward and discounted future value
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += transition[1]*(self.mdp.getReward(state, action,
                                                        transition[0]) + self.discount * self.values[transition[0]])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Return None if the state is terminal
        if self.mdp.isTerminal(state):
            return None
        else:
            maxValue = -float('inf')
            bestAction = None

            # Iterate over all possible actions for the current state
            for action in self.mdp.getPossibleActions(state):
                # Calculate the Q-value for the current state-action pair
                qValue = self.computeQValueFromValues(state, action)
                # Update the maximum value with the highest value found among all possible actions
                if qValue > maxValue:
                    maxValue = qValue
                    bestAction = action

        return bestAction

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        for i in range(self.iterations):
            # Get the state to update
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]

            # Update the value of the state if it is not terminal
            if self.mdp.isTerminal(state):
                self.values[state] = 0
            else:
                possibleActions = self.mdp.getPossibleActions(state)
                qValues = []

                for action in possibleActions:
                    # Get the transition states and probabilities for the current state-action pair
                    transitions = self.mdp.getTransitionStatesAndProbs(
                        state, action)
                    value = 0

                    # Calculate the value for the next state by discounting the future rewards
                    for next_state, probability in transitions:
                        reward = self.mdp.getReward(state, action, next_state)
                        value += probability * \
                            (reward + self.discount * self.values[next_state])

                    # Add the Q-value to the list of Q-values
                    qValues.append(value)

                # Update the value of the state with the maximum Q-value
                self.values[state] = max(qValues)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {}
        pQueue = util.PriorityQueue()

        # Compute predecessors for each state
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                        predecessors[nextState].add(state)

        # Compute differences and push into the priority queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                qValues = []
                for action in self.mdp.getPossibleActions(state):
                    qValues.append(self.computeQValueFromValues(state, action))
                diff = abs(self.values[state] - max(qValues))
                # Push the negative difference into the priority queue
                pQueue.update(state, -diff)

        # Perform value iteration updates
        for i in range(self.iterations):
            if pQueue.isEmpty():
                break
            state = pQueue.pop()
            if not self.mdp.isTerminal(state):
                qValues = []
                for action in self.mdp.getPossibleActions(state):
                    qValues.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(qValues)

            if state in predecessors:
                for p in predecessors[state]:
                    if not self.mdp.isTerminal(p):
                        qValues = []
                        for action in self.mdp.getPossibleActions(p):
                            qValues.append(
                                self.computeQValueFromValues(p, action))
                        diff = abs(self.values[p] - max(qValues))

                        if diff > self.theta:
                            pQueue.update(p, -diff)
