# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discountRate (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.values = util.Counter() #will store all Q(s,a) values in this list


  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    if action != None:
      return self.values[(state,action)]
    else:
      return 0.0


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    bestAction = self.getPolicy(state)
    if bestAction != None:
      return self.getQValue(state, bestAction)
    else:
      return 0.0


  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    moves = []
    choices = []
    for candidate in self.getLegalActions(state):
      moves.append((self.getQValue(state,candidate), candidate))

    if len(moves) != 0:
      #gets highest values and then finds all actions that have this value
      bestMove = max(moves)
      for value, move in moves:
        if value == bestMove[0]:
          #these actions then get put in 'choices' where one action is
          # randomly chosen
          choices.append(move)
      return random.choice(choices)

    else:
      return None


  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return None

    #chooses random action epsilon percent of time and optimal
    #action for the remaining percent of time
    if util.flipCoin(self.epsilon) == True:
      return random.choice(legalActions)
    else:
      return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    newQ = self.getValue(nextState)
    newValue = reward + self.discountRate*newQ
    alpha = self.alpha
    #Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample
    #sample = reward + discount rate * max_Action Q(s',a')
    self.values[(state,action)] = (1-alpha)*self.values[(state,action)]\
                                  + alpha* newValue
    return


class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)
    self.weights = util.Counter()
    # You might want to initialize weights here.

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    features = SimpleExtractor().getFeatures(state,action)
    totalValue = 0.0
    #Q(s,a) = summation from i to n of Fi(s,a)*Wi
    for feature in features.items():
      #feature[0] = is a feature, feature[1] = its associated value
      value = self.weights[feature[0]] * feature[1]
      totalValue += value
    return totalValue


  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    #newW = Q(s,a), newValue = correction, alpha = learning rate
    features = SimpleExtractor().getFeatures(state, action)
    newW = self.getQValue(state,action)
    newValue = (reward + self.discountRate*QLearningAgent().
                getValue(nextState)) - newW
    alpha = self.alpha

    #Wi = Wi + alpha * correction * Fi(s,a)
    #correction = (reward + discount*V(s')) - Q(s,a)
    for feature in features.items():
      self.weights[feature[0]] += alpha*newValue*feature[1]

    return


  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      util.raiseNotDefined()
