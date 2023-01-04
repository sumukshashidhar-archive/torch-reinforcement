"""
Level 0 Game. XOR Response Bit
Author: Sumuk Shashidhar (sumuk [at] sumuk.org)
"""
import typing as t
from random import randint
class ResponseBit:

    def __init__(self) -> None:
        """
        Initialize the ResponseBit Game

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # set the value of the bit
        self.value = randint(0, 1)

    def get_value(self) -> int:
        """
        Get the value of the bit

        Parameters
        ----------
        None

        Returns
        -------
        int
            The value of the bit
        """
        return self.value
    
    def take_response(self, response: int) -> bool:
        """
        Take a response from the player and returns whether or not you won

        Parameters
        ----------
        response : int
            The response from the player

        Returns
        -------
        bool
            Whether you won or not
        """
        # check if the response is correct
        return response != self.value


if __name__ == "__main__":
    rb = ResponseBit()
    print(rb.get_value())
    
