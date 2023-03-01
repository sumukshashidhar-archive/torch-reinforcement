"""
A simple game
"""

from random import randint

class BaselineGame:
    """
    The baseline game's goal is simple. Accumulate as many points as possible (upto 500).

    You get a point if you match the number that the computer has chosen.

    If you get it wrong, your streak resets.
    """

    def __init__(self) -> None:
        """
        Initialize the BaselineGame with 0 score and a random number
        """
        self.score = 0
        self.number = randint(0, 1)
    
    def get_response(self, num: int) -> bool:
        """
        Once we get a response, we check if it is correct or not.

        We update the score accordingly, and update the number as well.
        """
        n = self.number
        self.number = randint(0, 1)
        if num == n:
            self.score += 1
            return True
        else:
            self.score = 0
            return False


    def play_step(self, action: int) -> tuple:
        """
        Play a step in the game

        Parameters
        ----------
        action : int
            The action to take

        Returns
        -------
        tuple
            The reward, whether the game is over, and the score
        """
        response = self.get_response(action)
        if response:
            if self.score >= 500:
                return 1, True, self.score
            else:
                return 1, False, self.score
        else:
            return -1, True, self.score
        