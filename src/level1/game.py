"""
This simple game is a demonstration of deep Q reinforcement learning.
"""
class JumpGame:

    def __init__(self):
        # the game state initially is where the player is standing
        # completely away from the goal
        self.array = ['P', 'X', 'X', 'X', 'X', 'G']
        # the pointer is the index of the player
        self.pointer = 0
        # we also start off with a score
        self.score = 0
        # we also have a max moves counter
        # if you spend more than 20 moves to reach the goal, you just die
        self.max_moves = 20
    

    # move right
    def move_right(self):
        # first, we need to check if we can move right
        if self.pointer == len(self.array) - 1:
            # we cannot move right
            pass
        else:
            # we push the player to the right
            self.array[self.pointer], self.array[self.pointer + 1] = 'X', 'P'
            self.pointer += 1
        return
    
    # move left
    def move_left(self):
        # first, we need to check if we can move left
        if self.pointer == 0:
            # we cannot move left
            pass
        else:
            # we push the player to the left
            self.array[self.pointer], self.array[self.pointer - 1] = 'X', 'P'
            self.pointer -= 1
        return

    def check_goal(self):
        return self.pointer == len(self.array) - 1

    def reset(self):
        self.array = ['P', 'X', 'X', 'X', 'X', 'G']
        self.pointer = 0
        self.score = 0
        self.max_moves = 20
        return
    
    # now, we need to define the movements
    def play_step(self, action):
        self.max_moves -= 1
        # in this case, the action will be a tuple of len 2
        # like [0, 0] -> no movement
        # [0, 1] -> move right
        # [1, 0] -> move left
        if action == (0, 0):
            pass
        elif action == (0, 1):
            self.move_right()
        elif action == (1, 0):
            self.move_left()
        
        # we need to check if we have reached the goal
        if self.check_goal():
            # we need to tell the agent that it has reached the goal
            # this is the game over state
            return 1000, True, self.score
        elif self.max_moves == 0:
            # this means the the agent has spent too much time
            return -1000, True, self.score



