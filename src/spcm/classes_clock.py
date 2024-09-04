# -*- coding: utf-8 -*-

from .constants import *

from . import units

from .classes_functionality import CardFunctionality 
from .classes_unit_conversion import UnitConversion

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
    
    def max_sample_rate(self, return_unit = None) -> int:
        """
        Returns the maximum sample rate of the active card (see register `SPC_MIINST_MAXADCLOCK` in the manual)
    
        Returns
        -------
        int
        """
        
        max_sr = self.card.get_i(SPC_MIINST_MAXADCLOCK)
        if return_unit is not None: max_sr = UnitConversion.to_unit(max_sr * units.Hz, return_unit)
        return max_sr

    def sample_rate(self, sample_rate = 0, max : bool = False, return_unit = None) -> int:
        """
        Sets or gets the current sample rate of the handled card (see register `SPC_SAMPLERATE` in the manual)

        Parameters
        ----------
        sample_rate : int | pint.Quantity = 0
            if the parameter sample_rate is given with the function call, then the card's sample rate is set to that value
        max : bool = False
            if max is True, the method sets the maximum sample rate of the card
        unit : pint.Unit = None
            the unit of the sample rate, by default None
    
        Returns
        -------
        int
            the current sample rate in Samples/s
        """
        
        if max: sample_rate = self.max_sample_rate()
        if sample_rate:
            if isinstance(sample_rate, units.Quantity) and sample_rate.check("[]"):
                max_sr = self.max_sample_rate()
                sample_rate = sample_rate.to_base_units().magnitude * max_sr
            sample_rate = UnitConversion.convert(sample_rate, units.Hz, int)
            self.card.set_i(SPC_SAMPLERATE, int(sample_rate))
        return_value = self.card.get_i(SPC_SAMPLERATE)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.Hz, return_unit)
        return return_value
    
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
            self.card.set_i(SPC_CLOCKOUT, int(clock_output))
        return self.card.get_i(SPC_CLOCKOUT)
    output = clock_output
    
    def reference_clock(self, reference_clock : int = None) -> int:
        """
        Set the reference clock of the card (see register `SPC_REFERENCECLOCK` in the manual)
        
        Parameters
        ----------
        reference_clock : int | pint.Quantity
            the reference clock of the card in Hz
        
        Returns
        -------
        int
            the reference clock of the card in Hz
        """
        
        if reference_clock is not None:
            reference_clock = UnitConversion.convert(reference_clock, units.Hz, int)
            self.card.set_i(SPC_REFERENCECLOCK, reference_clock)
        return self.card.get_i(SPC_REFERENCECLOCK)