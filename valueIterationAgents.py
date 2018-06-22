# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discountRate = 0.9, iters = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discountRate = discountRate
    self.iters = iters
    self.values = util.Counter() # A Counter is a dict with default 0
    self.newValues = util.Counter() #where we will store the updated list of values

    i = 0
    while i <= self.iters:
      #first we initialize/update before starting next iteration
      for pairs in self.newValues.items():
        self.values[pairs[0]] = pairs[1]

      #V(s) = max_Action Q(s,a)
      for state in self.mdp.getStates():
        bestAction = self.getAction(state)
        value = self.getQValue(state,bestAction)
        self.newValues[state] = value

      i += 1

    return


  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    if action != None:
      successors = self.mdp.getTransitionStatesAndProbs(state, action)

      qValue = 0
      #this for loop will allow us to get the summation of the Bellman equation
      for nextState in successors:
        currReward = self.mdp.getReward(state, action, nextState[0])
        nextValue = self.getValue(nextState[0])
        #next V(s) = sum of transition states * reward * discount rate * V(s')
        value = nextState[1]*(currReward + self.discountRate*nextValue)
        qValue += value

      return qValue

    else:
      return 0


  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    moves = []
    for candidate in self.mdp.getPossibleActions(state):
      moves.append((self.getQValue(state,candidate), candidate))

    if len(moves) != 0:
      #finds a max and returns that associated action
      bestMove = max(moves)
      return bestMove[1]
    else:
      return None


  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
