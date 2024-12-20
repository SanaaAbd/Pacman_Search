# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    """
    # Initialize a stack to store the fringe (frontier) nodes
    pF = util.Stack()

    # Create the initial node: (state, path, total_cost)
    cN = (problem.getStartState(), [], [])

    # Push the initial node onto the fringe
    pF.push(cN)

    # Initialize a list to keep track of closed (visited) nodes
    closedNodes = []

    # Continue searching while there are nodes in the fringe
    while not pF.isEmpty():
        # Pop the next node from the fringe
        fringeNode, searchPath, fringeTotal = pF.pop()

        # If this node is the goal state, return the path to reach it
        if problem.isGoalState(fringeNode):
            return searchPath

        # If this node hasn't been visited yet
        if fringeNode not in closedNodes:
            # Mark it as visited
            closedNodes.append(fringeNode)

            # Explore all successors of this node
            for newCoords, newMove, newCost in problem.getSuccessors(fringeNode):
                # Push each successor onto the fringe
                pF.push((newCoords, searchPath + [newMove], fringeTotal + [newCost]))

def breadthFirstSearch(problem):
    # Initialize a queue for BFS
    queue = util.Queue()

    # Initialize a set to keep track of visited states
    visited = set()

    # Get the initial state from the problem
    start_state = problem.getStartState()

    # Add the start state to the queue along with an empty action list
    queue.push((start_state, []))

    # Main BFS loop
    while not queue.isEmpty():
        # Get the next state and actions from the queue
        current_state, actions = queue.pop()

        # Check if we've reached the goal state
        if problem.isGoalState(current_state):
            return actions  # Return the list of actions to reach the goal

        # If this state hasn't been visited yet
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)

            # Get all possible successors of the current state
            for successor, action, _ in problem.getSuccessors(current_state):
                # If the successor hasn't been visited
                if successor not in visited:
                    # Create a new action list by appending the new action
                    new_actions = actions + [action]
                    # Add the successor and its action list to the queue
                    queue.push((successor, new_actions))

    # If no solution is found, return an empty list
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    # Initialize a priority queue for the fringe
    pacFringe = util.PriorityQueue()

    # Initialize a counter to keep track of costs
    pacCount = util.Counter()

    # Create the initial node: (state, path)
    currentNode = (problem.getStartState(), [])

    # Initialize a list to keep track of closed (visited) nodes
    closedNodes = []

    # Push the initial node onto the fringe with priority 0
    pacFringe.push(currentNode, 0)

    # Continue searching while there are nodes in the fringe
    while not pacFringe.isEmpty():
        # Pop the node with the lowest cost from the fringe
        fringeNode, searchPath = pacFringe.pop()

        # If this node is the goal state, return the path to reach it
        if problem.isGoalState(fringeNode):
            return searchPath

        # If this node hasn't been visited yet
        if fringeNode not in closedNodes:
            # Mark it as visited
            closedNodes.append(fringeNode)

            # Explore all successors of this node
            for newCoords, newMove, newCost in problem.getSuccessors(fringeNode):
                # Calculate the cost to reach this successor
                pacCount[newCoords] = pacCount[fringeNode] + newCost

                # Push the successor onto the fringe with its total cost as priority
                pacFringe.push((newCoords, searchPath + [newMove]), pacCount[newCoords])

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # Initialize a priority queue for the frontier
    frt = util.PriorityQueue()

    # Initialize a list to keep track of visited nodes and their costs
    visitedList = []

    # Push the starting state into the frontier queue
    # Format: (state, path, cost), priority
    # Priority is f(n) = g(n) + h(n), where g(n) is the cost so far (0 initially) and h(n) is the heuristic estimate
    frt.push((problem.getStartState(), [], 0), 0 + heuristic(problem.getStartState(), problem))

    # Pop the first node from the frontier
    (state, toDirection, toCost) = frt.pop()

    # Add the initial state to the visited list along with its f(n) value
    visitedList.append((state, toCost + heuristic(problem.getStartState(), problem)))

    # Continue searching until we reach the goal state
    while not problem.isGoalState(state):
        # Get all successors of the current state
        successors = problem.getSuccessors(state)

        # Examine each successor
        for son in successors:
            visitedExist = False
            total_cost = toCost + son[2]  # g(n) for this successor

            # Check if this successor has been visited before with a lower or equal cost
            for (visitedState, visitedToCost) in visitedList:
                if (son[0] == visitedState) and (total_cost >= visitedToCost):
                    visitedExist = True
                    break

            # If this is a new state or has a lower cost than previously visited
            if not visitedExist:
                # Push the successor to the frontier
                # Priority is f(n) = g(n) + h(n)
                frt.push((son[0], toDirection + [son[1]], total_cost),
                         total_cost + heuristic(son[0], problem))

                # Add this successor to the visited list
                visitedList.append((son[0], total_cost))

        # Pop the next node with the lowest f(n) from the frontier
        (state, toDirection, toCost) = frt.pop()

    # If we've exited the while loop, we've reached the goal state
    # Return the path to the goal
    return toDirection

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
