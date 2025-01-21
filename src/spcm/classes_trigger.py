# -*- coding: utf-8 -*-

import numpy as np

from spcm_core.constants import *

from .classes_functionality import CardFunctionality
from .classes_card import Card
from .classes_channels import Channels, Channel

from .classes_unit_conversion import UnitConversion
from . import units
import pint

class Trigger(CardFunctionality):
    """a higher-level abstraction of the CardFunctionality class to implement the Card's Trigger engine"""

    channels : Channels = None

    def __init__(self, card : 'Card', **kwargs) -> None:
        """
        Constructor of the Trigger class
        
        Parameters
        ----------
        card : Card
            The card to use for the Trigger class
        """

        super().__init__(card)
        self.channels = kwargs.get('channels', None)
    
    def __str__(self) -> str:
        """
        String representation of the Trigger class
    
        Returns
        -------
        str
            String representation of the Trigger class
        """
        
        return f"Trigger(card={self.card})"
    
    __repr__ = __str__

    def enable(self) -> None:
        """Enables the trigger engine (see command 'M2CMD_CARD_ENABLETRIGGER' in chapter `Trigger` in the manual)"""
        self.card.cmd(M2CMD_CARD_ENABLETRIGGER)
    
    def disable(self) -> None:
        """Disables the trigger engine (see command 'M2CMD_CARD_DISABLETRIGGER' in chapter `Trigger` in the manual)"""
        self.card.cmd(M2CMD_CARD_DISABLETRIGGER)
    
    def force(self) -> None:
        """Forces a trigger event if the hardware is still waiting for a trigger event. (see command 'M2CMD_CARD_FORCETRIGGER' in chapter `Trigger` in the manual)"""
        self.card.cmd(M2CMD_CARD_FORCETRIGGER)
    
    def write_setup(self) -> None:
        """Write the trigger setup to the card"""
        self.card.write_setup()
    
    # OR Mask
    def or_mask(self, mask : int = None) -> int:
        """
        Set the OR mask for the trigger input lines (see register 'SPC_TRIG_ORMASK' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mask : int
            The OR mask for the trigger input lines
        
        Returns
        -------
        int
            The OR mask for the trigger input lines
        """

        if mask is not None:
            self.card.set_i(SPC_TRIG_ORMASK, mask)
        return self.card.get_i(SPC_TRIG_ORMASK)

    # AND Mask
    def and_mask(self, mask : int = None) -> int:
        """
        Set the AND mask for the trigger input lines (see register 'SPC_TRIG_ANDMASK' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mask : int
            The AND mask for the trigger input lines
        
        Returns
        -------
        int
            The AND mask for the trigger input lines
        """

        if mask is not None:
            self.card.set_i(SPC_TRIG_ANDMASK, mask)
        return self.card.get_i(SPC_TRIG_ANDMASK)

    # Channel triggering
    def ch_mode(self, channel, mode : int = None) -> int:
        """
        Set the mode for the trigger input lines (see register 'SPC_TRIG_CH0_MODE' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int | Channel
            The channel to set the mode for
        mode : int
            The mode for the trigger input lines
        
        Returns
        -------
        int
            The mode for the trigger input lines
        
        """

        channel_index = int(channel)
        if mode is not None:
            self.card.set_i(SPC_TRIG_CH0_MODE + channel_index, mode)
        return self.card.get_i(SPC_TRIG_CH0_MODE + channel_index)

    def ch_level(self, channel : int, level_num : int, level_value = None, return_unit : pint.Unit = None) -> int:
        """
        Set the level for the trigger input lines (see register 'SPC_TRIG_CH0_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int | Channel
            The channel to set the level for
        level_num : int
            The level 0 or level 1
        level_value : int | pint.Quantity | None
            The level for the trigger input lines
        
        Returns
        -------
        int
            The level for the trigger input lines
        """

        channel_index = int(channel)
        # if a level value is given in the form of a quantity, convert it to the card's unit as a integer value
        if isinstance(level_value, units.Quantity):
            if isinstance(channel, Channel):
                level_value = channel.reconvert_data(level_value)
            elif self.channels and isinstance(self.channels[channel_index], Channel):
                level_value = self.channels[channel_index].reconvert_data(level_value)
            else:
                raise ValueError("No channel information available to convert the trigger level value. Please provide a channel object or set the channel information in the Trigger object.")
        
        if isinstance(level_value, int):
            self.card.set_i(SPC_TRIG_CH0_LEVEL0 + channel_index + 100 * level_num, level_value)

        return_value = self.card.get_i(SPC_TRIG_CH0_LEVEL0 + channel_index + 100 * level_num)
        # if a return unit is given, convert the value to the given unit if a channel object is available
        if isinstance(return_unit, pint.Unit):
            if isinstance(channel, Channel):
                return_value = channel.convert_data(return_value, return_unit=return_unit)
            elif self.channels and isinstance(self.channels[channel_index], Channel):
                return_value = self.channels[channel_index].convert_data(return_value, return_unit=return_unit)
            else:
                raise ValueError("No channel information available to convert the returning trigger level value. Please provide a channel object or set the channel information in the Trigger object.")
            
        return return_value

    def ch_level0(self, channel : int, level_value = None, return_unit : pint.Unit = None) -> int:
        """
        Set the level 0 for the trigger input lines (see register 'SPC_TRIG_CH0_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int | Channel
            The channel to set the level for
        level_value : int | pint.Quantity | None
            The level for the trigger input lines
        
        Returns
        -------
        int
            The level for the trigger input lines
        """

        return self.ch_level(channel, 0, level_value, return_unit)
    
    def ch_level1(self, channel : int, level_value = None, return_unit : pint.Unit = None) -> int:
        """
        Set the level 1 for the trigger input lines (see register 'SPC_TRIG_CH0_LEVEL1' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int | Channel
            The channel to set the level for
        level_value : int | pint.Quantity | None
            The level for the trigger input lines
        
        Returns
        -------
        int
            The level for the trigger input lines
        """

        return self.ch_level(channel, 1, level_value, return_unit)

    # Channel OR Mask0
    def ch_or_mask0(self, mask : int = None) -> int:
        """
        Set the channel OR mask0 for the trigger input lines (see register 'SPC_TRIG_CH_ORMASK0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mask : int
            The OR mask for the trigger input lines
        
        Returns
        -------
        int
            The OR mask for the trigger input lines
        """

        if mask is not None:
            self.card.set_i(SPC_TRIG_CH_ORMASK0, mask)
        return self.card.get_i(SPC_TRIG_CH_ORMASK0)
    
    # Channel AND Mask0
    def ch_and_mask0(self, mask : int = None) -> int:
        """
        Set the AND mask0 for the trigger input lines (see register 'SPC_TRIG_CH_ANDMASK0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mask : int
            The AND mask0 for the trigger input lines
        
        Returns
        -------
        int
            The AND mask0 for the trigger input lines
        """

        if mask is not None:
            self.card.set_i(SPC_TRIG_CH_ANDMASK0, mask)
        return self.card.get_i(SPC_TRIG_CH_ANDMASK0)
    
    # Delay
    def delay(self, delay = None, return_unit : pint.Unit = None) -> int:
        """
        Set the delay for the trigger input lines in number of sample clocks (see register 'SPC_TRIG_DELAY' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        delay : int | pint.Quantity
            The delay for the trigger input lines
        return_unit : pint.Unit
            The unit to return the value in

        Returns
        -------
        int | pint.Quantity
            The delay for the trigger input lines

        NOTE
        ----
        different cards have different step sizes for the delay. 
        If a delay with unit is given, this function takes the value, 
        calculates the integer value and rounds to the nearest allowed delay value
        """

        sr = self.card.get_i(SPC_SAMPLERATE) * units.Hz
        if delay is not None:
            if isinstance(delay, units.Quantity):
                delay_step = self.avail_delay_step()
                delay = np.rint(int(delay * sr) / delay_step).astype(np.int64) * delay_step
            self.card.set_i(SPC_TRIG_DELAY, delay)
        return_value = self.card.get_i(SPC_TRIG_DELAY)
        if isinstance(return_unit, pint.Unit): return_value = UnitConversion.to_unit(return_value / sr, return_unit)
        return return_value
    
    def avail_delay_max(self) -> int:
        """
        Get the maximum delay for the trigger input lines in number of sample clocks (see register 'SPC_TRIG_AVAILDELAY' in chapter `Trigger` in the manual)
        
        Returns
        -------
        int
            The maximum delay for the trigger input lines
        """

        return self.card.get_i(SPC_TRIG_AVAILDELAY)
    
    def avail_delay_step(self) -> int:
        """
        Get the step size for the delay for the trigger input lines in number of sample clocks (see register 'SPC_TRIG_AVAILDELAY_STEP' in chapter `Trigger` in the manual)
        
        Returns
        -------
        int
            The step size for the delay for the trigger input lines
        """

        return self.card.get_i(SPC_TRIG_AVAILDELAY_STEP)

    
    def trigger_counter(self) -> int:
        """
        Get the number of trigger events since acquisition start (see register 'SPC_TRIGGERCOUNTER' in chapter `Trigger` in the manual)
        
        Returns
        -------
        int
            The trigger counter
        """

        return self.card.get_i(SPC_TRIGGERCOUNTER)
    
    # Main external window trigger (ext0/Trg0)
    def ext0_mode(self, mode : int = None) -> int:
        """
        Set the mode for the main external window trigger (ext0/Trg0) (see register 'SPC_TRIG_EXT0_MODE' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mode : int
            The mode for the main external window trigger (ext0/Trg0)
        
        Returns
        -------
        int
            The mode for the main external window trigger (ext0/Trg0)
        """

        if mode is not None:
            self.card.set_i(SPC_TRIG_EXT0_MODE, mode)
        return self.card.get_i(SPC_TRIG_EXT0_MODE)
    
    # Trigger termination
    def termination(self, termination : int = None) -> int:
        """
        Set the trigger termination (see register 'SPC_TRIG_TERM' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        termination : int
            The trigger termination: a „1“ sets the 50 Ohm termination for external trigger signals. A „0“ sets the high impedance termination
        
        Returns
        -------
        int
            The trigger termination: a „1“ sets the 50 Ohm termination for external trigger signals. A „0“ sets the high impedance termination
        """

        if termination is not None:
            self.card.set_i(SPC_TRIG_TERM, termination)
        return self.card.get_i(SPC_TRIG_TERM)
    
    # Trigger input coupling
    def ext0_coupling(self, coupling : int = None) -> int:
        """
        Set the trigger input coupling (see hardware manual register name 'SPC_TRIG_EXT0_ACDC')
        
        Parameters
        ----------
        coupling : int
            The trigger input coupling: COUPLING_DC enables DC coupling, COUPLING_AC enables AC coupling for the external trigger 
            input (AC coupling is the default).

        Returns
        -------
        int
            The trigger input coupling: COUPLING_DC enables DC coupling, COUPLING_AC enables AC coupling for the external trigger 
            input (AC coupling is the default).
        """

        if coupling is not None:
            self.card.set_i(SPC_TRIG_EXT0_ACDC, coupling)
        return self.card.get_i(SPC_TRIG_EXT0_ACDC)
    
    # ext1 trigger mode
    def ext1_mode(self, mode : int = None) -> int:
        """
        Set the mode for the ext1 trigger (see register 'SPC_TRIG_EXT1_MODE' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        mode : int
            The mode for the ext1 trigger
        
        Returns
        -------
        int
            The mode for the ext1 trigger
        """

        if mode is not None:
            self.card.set_i(SPC_TRIG_EXT1_MODE, mode)
        return self.card.get_i(SPC_TRIG_EXT1_MODE)
    
    # Trigger level
    def ext0_level0(self, level = None, return_unit = None) -> int:
        """
        Set the trigger level 0 for the ext0 trigger (see register 'SPC_TRIG_EXT0_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level 0 for the ext0 trigger in mV
        return_unit : pint.Unit
            The unit to return the value in
        
        Returns
        -------
        int | pint.Quantity
            The trigger level 0 for the ext0 trigger in mV or in the specified unit
        """

        if level is not None:
            level = UnitConversion.convert(level, units.mV, int)
            self.card.set_i(SPC_TRIG_EXT0_LEVEL0, level)
        return_value = self.card.get_i(SPC_TRIG_EXT0_LEVEL0)
        return UnitConversion.to_unit(return_value * units.mV, return_unit)
    
    def ext0_level1(self, level = None, return_unit = None) -> int:
        """
        Set the trigger level 1 for the ext0 trigger (see register 'SPC_TRIG_EXT0_LEVEL1' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level for the ext0 trigger in mV
        return_unit : pint.Unit
            The unit to return the value in
        
        Returns
        -------
        int | pint.Quantity
            The trigger level for the ext0 trigger in mV or in the specified unit
        """

        if level is not None:
            level = UnitConversion.convert(level, units.mV, int)
            self.card.set_i(SPC_TRIG_EXT0_LEVEL1, level)
        return_value = self.card.get_i(SPC_TRIG_EXT0_LEVEL1)
        return UnitConversion.to_unit(return_value * units.mV, return_unit)
    
    def ext1_level0(self, level = None, return_unit = None) -> int:
        """
        Set the trigger level 0 for the ext1 trigger (see register 'SPC_TRIG_EXT1_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level 0 for the ext1 trigger in mV
        return_unit : pint.Unit
            The unit to return the value in
        
        Returns
        -------
        int | pint.Quantity
            The trigger level 0 for the ext1 trigger in mV or in the specified unit
        """

        if level is not None:
            level = UnitConversion.convert(level, units.mV, int)
            self.card.set_i(SPC_TRIG_EXT1_LEVEL0, level)
        return_value = self.card.get_i(SPC_TRIG_EXT1_LEVEL0)
        return UnitConversion.to_unit(return_value * units.mV, return_unit)