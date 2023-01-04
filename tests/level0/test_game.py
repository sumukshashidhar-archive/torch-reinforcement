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

    def test_get_value(self):
        """
        Test the get_value method of the ResponseBit class

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
        self.assertTrue(rb.get_value() == 0 or rb.get_value() == 1)
    
    def test_take_response(self):
        """
        Test the take_response method of the ResponseBit class

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the response is not 0 or 1
        """
        rb = game.ResponseBit()
        self.assertTrue(rb.take_response(0) == True or rb.take_response(1) == True)