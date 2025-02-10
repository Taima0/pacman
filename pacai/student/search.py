"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # *** Your Code Here ***

    from pacai.util.stack  import Stack

    stack = Stack()
    stack.push((problem.startingState(), []))
    visited = set()

    while not stack.isEmpty():
        state,actions = stack.pop()

        if  state in visited:
            continue

        visited.add(state)

        if problem.isGoal(state):
            return actions 

        for successor, action, cost, in problem.successorStates(state):
            if successor not in visited:
                new_actions = actions + [action]
                stack.push((successor, new_actions))
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***

    from pacai.util.queue  import Queue

    queue = Queue()
    queue.push((problem.startingState(), []))
    visited = set()

    while not queue.isEmpty():
        state,actions = queue.pop()


        if state in visited:
            continue

        visited.add(state)

        if problem.isGoal(state):
            return actions

        for successor, action, cost, in problem.successorStates(state):
            if successor not in visited:
                new_actions = actions + [action]
                queue.push((successor, new_actions))
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    from pacai.util.priorityQueue import PriorityQueue

    # *** Your Code Here ***

    priority_queue = PriorityQueue()
    priority_queue.push((problem.startingState(), [], 0), 0)
    visited = set()

    while not priority_queue.isEmpty():
        state,actions, cost = priority_queue.pop()

        if  state in visited:
            continue

        visited.add(state)

        if problem.isGoal(state):
            return actions

        for successor, action, dif_cost, in problem.successorStates(state):
            if successor not in visited:
                update_cost = cost + dif_cost
                new_actions = actions + [action]
                priority_queue.push((successor, new_actions, update_cost), update_cost)
    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    from pacai.util.priorityQueue import PriorityQueue

    # *** Your Code Here ***

    priority_queue = PriorityQueue()
    priority_queue.push((problem.startingState(), [], 0), 0)
    visited = set()

    while not priority_queue.isEmpty():
        state,actions, cost = priority_queue.pop()

        if  state in visited:
            continue

        visited.add(state)

        if problem.isGoal(state):
            return actions

        for successor, action, dif_cost, in problem.successorStates(state):
            if successor not in visited:
                update_cost = cost + dif_cost
                new_actions = actions + [action]
                priority = update_cost + heuristic(successor, problem)
                priority_queue.push((successor, new_actions, update_cost), heuristic)
    return []
