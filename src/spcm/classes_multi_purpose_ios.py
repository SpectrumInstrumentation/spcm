# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_card import Card
from .classes_functionality import CardFunctionality 

class MultiPurposeIO:
    """a higher-level abstraction of the CardFunctionality class to implement the Card's Multi purpose I/O functionality"""

    card : Card = None
    x_index : int = None

    def __init__(self, card : Card, x_index : int = None) -> None:
        """
        Constructor for the MultiPurposeIO class
    
        Parameters
        ----------
        card : Card
            The card object to communicate with the card
        x_index : int
            The index of the digital input/output to be enabled.
        """

        self.card = card
        self.x_index = x_index
    
    def __str__(self) -> str:
        """
        String representation of the MultiPurposeIO class
    
        Returns
        -------
        str
            String representation of the MultiPurposeIO class
        """
        
        return f"MultiPurposeIO(card={self.card}, x_index={self.x_index})"
    
    __repr__ = __str__

    def avail_modes(self) -> int:
        """
        Returns the available modes of the digital input/output of the card (see register 'SPCM_X0_AVAILMODES' in chapter `Multi Purpose I/O Lines` in the manual)

        Returns
        -------
        int
            The available modes of the digital input/output
        """

        return self.card.get_i(SPCM_X0_AVAILMODES + self.x_index)

    def x_mode(self, mode : int = None) -> int:
        """
        Sets the mode of the digital input/output of the card (see register 'SPCM_X0_MODE' in chapter `Multi Purpose I/O Lines` in the manual)
    
        Parameters
        ----------
        mode : int
            The mode of the digital input/output
        
        Returns
        -------
        int
            The mode of the digital input/output
        """

        if mode is not None:
            self.card.set_i(SPCM_X0_MODE + self.x_index, mode)
        return self.card.get_i(SPCM_X0_MODE + self.x_index)
    
    def dig_mode(self, mode : int = None) -> int:
        """
        Sets the digital input/output mode of the xio line (see register 'SPCM_DIGMODE0' in chapter `Multi Purpose I/O Lines` in the manual)
    
        Parameters
        ----------
        mode : int
            The digital input/output mode of the xio line

        Returns
        -------
        int
            The digital input/output mode of the xio line
        """

        if mode is not None:
            self.card.set_i(SPC_DIGMODE0 + self.x_index, mode)
        return self.card.get_i(SPC_DIGMODE0 + self.x_index)

class MultiPurposeIOs(CardFunctionality):
    """a higher-level abstraction of the CardFunctionality class to implement the Card's Multi purpose I/O functionality"""

    xio_lines : list[MultiPurposeIO] = []
    num_xio_lines : int = None

    def __init__(self, card : Card, *args, **kwargs) -> None:
        """
        Constructor for the MultiPurposeIO class
    
        Parameters
        ----------
        card : Card
            The card object to communicate with the card
        """

        super().__init__(card, *args, **kwargs)
        
        self.xio_lines = []
        self.num_xio_lines = self.get_num_xio_lines()
        self.load()
    
    def __str__(self) -> str:
        """
        String representation of the MultiPurposeIO class
    
        Returns
        -------
        str
            String representation of the MultiPurposeIO class
        """
        
        return f"MultiPurposeIOs(card={self.card})"
    
    __repr__ = __str__
    def __iter__(self) -> "MultiPurposeIOs":
        """Define this class as an iterator"""
        return self
    
    def __getitem__(self, index : int) -> MultiPurposeIO:
        """
        Get the xio line at the given index

        Parameters
        ----------
        index : int
            The index of the xio line to be returned
        
        Returns
        -------
        MultiPurposeIO
            The xio line at the given index
        """

        return self.xio_lines[index]
    
    _xio_iterator_index = -1
    def __next__(self) -> MultiPurposeIO:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        MultiPurposeIO
            The next xio line in the iterator
        
        Raises
        ------
        StopIteration
        """
        self._xio_iterator_index += 1
        if self._xio_iterator_index >= len(self.xio_lines):
            self._xio_iterator_index = -1
            raise StopIteration
        return self.xio_lines[self._xio_iterator_index]
    
    def __len__(self) -> int:
        """Returns the number of available xio lines of the card"""
        return len(self.xio_lines)
    

    def get_num_xio_lines(self) -> int:
        """
        Returns the number of digital input/output lines of the card (see register 'SPCM_NUM_XIO_LINES' in chapter `Multi Purpose I/O Lines` in the manual)
    
        Returns
        -------
        int
            The number of digital input/output lines of the card

        """

        return self.card.get_i(SPC_NUM_XIO_LINES)

    def load(self) -> None:
        """
        Loads the digital input/output lines of the card
        """

        self.xio_lines = [MultiPurposeIO(self.card, x_index) for x_index in range(self.num_xio_lines)]
    
    def asyncio(self, output : int = None) -> int:
        """
        Sets the async input/output of the card (see register 'SPCM_XX_ASYNCIO' in chapter `Multi Purpose I/O Lines` in the manual)
    
        Parameters
        ----------
        output : int
            The async input/output of the card

        Returns
        -------
        int
            The async input/output of the card
        """

        if output is not None:
            self.card.set_i(SPCM_XX_ASYNCIO, output)
        return self.card.get_i(SPCM_XX_ASYNCIO)
