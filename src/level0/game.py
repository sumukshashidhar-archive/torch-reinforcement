"""
Level 0 Game. XOR Response Bit
Author: Sumuk Shashidhar (sumuk [at] sumuk.org)
"""
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
        # set the score of the game
        self.score = 0

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
        if response != self.value:
            # if it is, we update the "score"
            self.score += 1
            # reset the value of the bit
            self.value = randint(0, 1)
            # and return True
            return True
        else:
            # if it isn't, we return False
            return False


if __name__ == "__main__":
    rb = ResponseBit()
    while True:
        print(rb.get_value())
        res = rb.take_response(int(input("Enter your response: ")))
        if res:
            print(f"You won! Your score is now {rb.score} :D. Let's continue.")
        else:
            print("You Lost! Game Over :(")

