import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
       
        score = successorGameState.getScore()
        foodList = newFood.asList()
        if len(foodList) > 0:
            minFoodDistance = min([manhattan(newPosition, food) for food in foodList])
            score += 10 / (minFoodDistance + 1)

            for ghostState in newGhostStates:
                ghostPos = ghostState.getPosition()
                ghostDist = manhattan(newPosition, ghostPos)
                if ghostState.getScaredTimer() > 0:
                    score += 200 / (ghostDist + 1)
                elif ghostDist < 2:
                    score -= 1000
            return score
        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***

        return successorGameState.getScore()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):

        numberGhosts = gameState.getNumAgents() - 1

        def maxV(state, depth, i):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            score = float('-inf')
            action = None
            moreActions = state.getLegalActions(i)

            for a in moreActions:
                successor = state.generateSuccessor(i, a)
                minScore = minV(successor, depth, 1)

                if minScore > score:
                    score = minScore
                    action = a

            return action if depth == self.getTreeDepth() else score

        def minV(state, depth, i):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            score = float('inf')
            moreActions = state.getLegalActions(i)

            for a in moreActions:
                successor = state.generateSuccessor(i, a)

                if i == numberGhosts:
                    maxScore = maxV(successor, depth - 1, 0)
                else:
                    maxScore = minV(successor, depth, i + 1)

                score = min(maxScore, score)
            return maxScore

        return maxV(gameState, self.getTreeDepth(), 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        '''
        numberGhosts = gameState.getNumAgents() -1
        '''

        def maxV(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            score = float('-inf')
            action = None
            moreActions = state.getLegalActions(0)

            for a in moreActions:
                successor = state.generateSuccessor(0, a)
                minScore = minV(successor, depth, 1, alpha, beta)

                if minScore > score:
                    score = minScore
                    action = a

                if score >= beta:
                    return score

            return action if depth == self.getTreeDepth() else score

        def minV(state, depth, i, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            score = float('inf')
            moreActions = state.getLegalActions(i)

            for a in moreActions:
                successor = state.generateSuccessor(i, a)

                if i == state.getNumAgents() - 1:
                    maxScore = maxV(successor, depth - 1, alpha, beta)
                else:
                    maxScore = minV(successor, depth, i + 1, alpha, beta)

                score = min(maxScore, score)

                if score <= alpha:
                    return score
                beta = min(beta, score)

            return score

        return maxV(gameState, self.getTreeDepth(), float(' -inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):

        def maxV(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            score = float('-inf')
            action = None
            moreActions = state.getLegalActions(0)

            for a in moreActions:
                successor = state.generateSuccessor(0, a)
                minScore = expV(successor, depth, 1)

                if minScore > score:
                    score = minScore
                    action = a
            return action if depth == self.getTreeDepth() else score

        def expV(state, depth, i):
            if state.isWin() or state.isLose() or depth == 0:
                return self.getEvaluationFunction()(state)

            moreActions = state.getLegalActions(i)

            if not moreActions:
                return self.getEvaluationFunction()(state)

            numActions = len(moreActions)
            expVal = 0
            for a in moreActions:
                successor = state.generateSuccessor(i, a)
                if i == state.getNumAgents() - 1:
                    expVal += maxV(successor, depth - 1)
                else:
                    expVal += expV(successor, depth, i + 1)
            return expVal / numActions
        return maxV(gameState, self.getTreeDepth())

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    caps = currentGameState.getCapsules()
    score = currentGameState.getScore()
    foodList = newFood.asList()

    foodMore = 0
    if len(foodList) > 0:
        minFoodDistance = min([manhattan(newPosition, food) for food in foodList])
        foodMore += 10 / (minFoodDistance + 1)

    capsMore = 0
    if caps:
        minCaps = min(manhattan(newPosition, cap) for cap in caps)
        capsMore = 50 / (minCaps + 1)

    ghostLess = 0
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattan(newPosition, ghostPos)

        if ghostState.getScaredTimer() > 0:
            score += 200 / (ghostDist + 1)
        elif ghostDist < 2:
            ghostLess += 1000

    foodLess = len(foodList) * 5

    return score + foodMore + capsMore - ghostLess - foodLess


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
