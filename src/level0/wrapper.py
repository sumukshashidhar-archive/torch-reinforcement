"""
This module wraps around the game, to make it interactable with the Agent

Author: Sumuk Shashidhar (sumuk [at] sumuk.org)
"""
from game import ResponseBit
class ResponseBitWrapper():

    def __init__(self) -> None:
        """
        Initialize the ResponseBitWrapper

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.game = ResponseBit()
        self.iterations = 0
    

    def reset(self) -> None:
        """
        Reset the game

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.game = ResponseBit()
        self.iterations += 1

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
            The reward, the next state, and whether the game is over
        """
        response = self.game.take_response(action)
        self.iterations += 1
        if response and self.iterations < 1000:
            # means that we won this round
            return 1, False, self.game.score
        elif response:
            # means that we won the game. Yay!
            return 1, True, self.game.score
        else:
            # means that we lost this round
            return -1, True, self.game.score

