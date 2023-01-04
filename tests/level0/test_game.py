import unittest
from src.level0 import game

class TestResponseBit(unittest.TestCase):

    def test_init(self):
        """
        Test the initialization of the ResponseBit class

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the value of the bit is not 0 or 1
        """
        rb = game.ResponseBit()
        self.assertTrue(rb.value == 0 or rb.value == 1)