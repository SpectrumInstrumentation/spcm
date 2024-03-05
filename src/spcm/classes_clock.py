# -*- coding: utf-8 -*-

from .constants import *

from .classes_functionality import CardFunctionality 

class Clock(CardFunctionality):
    """a higher-level abstraction of the CardFunctionality class to implement the Card's clock engine"""
    
    def __str__(self) -> str:
        """
        String representation of the Clock class
    
        Returns
        -------
        str
            String representation of the Clock class
        """
        
        return f"Clock(card={self.card})"
    
    __repr__ = __str__
    
    def write_setup(self) -> None:
        """Write the setup to the card"""
        self.card.write_setup()
    
    
    def mode(self, mode : int = None) -> int:
        """
        Set the clock mode of the card (see register `SPC_CLOCKMODE` in the manual)
    
        Parameters
        ----------
        mode : int
            The clock mode of the card
        
        Returns
        -------
        int
            The clock mode of the card
        """

        if mode is not None:
            self.card.set_i(SPC_CLOCKMODE, mode)
        return self.card.get_i(SPC_CLOCKMODE)
    
    def max_sample_rate(self) -> int:
        """
        Returns the maximum sample rate of the active card (see register `SPC_MIINST_MAXADCLOCK` in the manual)
    
        Returns
        -------
        int
        """
        
        return self.card.get_i(SPC_MIINST_MAXADCLOCK)

    def sample_rate(self, sample_rate : int = 0, max : bool = False) -> int:
        """
        Sets or gets the current sample rate of the handled card (see register `SPC_SAMPLERATE` in the manual)

        Parameters
        ----------
        sample_rate : int = 0
            if the parameter sample_rate is given with the function call, then the card's sample rate is set to that value
        max : bool = False
            if max is True, the method sets the maximum sample rate of the card
    
        Returns
        -------
        int
            the current sample rate in Samples/s
        """
        
        if max:
            sample_rate = self.max_sample_rate()
        if sample_rate:
            self.card.set_i(SPC_SAMPLERATE, int(sample_rate))
        return self.card.get_i(SPC_SAMPLERATE)
    
    def clock_output(self, clock_output : int = None) -> int:
        """
        Set the clock output of the card (see register `SPC_CLOCKOUT` in the manual)
        
        Parameters
        ----------
        clock_output : int
            the clock output of the card
        
        Returns
        -------
        int
            the clock output of the card
        """
        
        if clock_output is not None:
            self.card.set_i(SPC_CLOCKOUT, clock_output)
        return self.card.get_i(SPC_CLOCKOUT)
    output = clock_output
    
    def reference_clock(self, reference_clock : int = None) -> int:
        """
        Set the reference clock of the card (see register `SPC_REFERENCECLOCK` in the manual)
        
        Parameters
        ----------
        reference_clock : int
            the reference clock of the card in Hz
        
        Returns
        -------
        int
            the reference clock of the card in Hz
        """
        
        if reference_clock is not None:
            self.card.set_i(SPC_REFERENCECLOCK, reference_clock)
        return self.card.get_i(SPC_REFERENCECLOCK)