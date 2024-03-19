# -*- coding: utf-8 -*-

from .constants import *

from .classes_functionality import CardFunctionality 

class Trigger(CardFunctionality):
    """a higher-level abstraction of the CardFunctionality class to implement the Card's Trigger engine"""
    
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
    def ch_mode(self, channel : int, mode : int = None) -> int:
        """
        Set the mode for the trigger input lines (see register 'SPC_TRIG_CH0_MODE' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int
            The channel to set the mode for
        mode : int
            The mode for the trigger input lines
        
        Returns
        -------
        int
            The mode for the trigger input lines
        """

        if mode is not None:
            self.card.set_i(SPC_TRIG_CH0_MODE + channel, mode)
        return self.card.get_i(SPC_TRIG_CH0_MODE + channel)

    def ch_level(self, channel : int, level : int, trigger_level = None) -> int:
        """
        Set the level for the trigger input lines (see register 'SPC_TRIG_CH0_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        channel : int
            The channel to set the level for
        level : int
            The level 0 or level 1
        trigger_level : int
            The level for the trigger input lines
        
        Returns
        -------
        int
            The level for the trigger input lines
        """

        if level is not None:
            self.card.set_i(SPC_TRIG_CH0_LEVEL0 + channel + 100 * level, trigger_level)
        return self.card.get_i(SPC_TRIG_CH0_LEVEL0 + channel + 100 * level)

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
    def delay(self, delay : int = None) -> int:
        """
        Set the delay for the trigger input lines in number of sample clocks (see register 'SPC_TRIG_DELAY' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        delay : int
            The delay for the trigger input lines

        Returns
        -------
        int
            The delay for the trigger input lines
        """

        if delay is not None:
            self.card.set_i(SPC_TRIG_DELAY, delay)
        return self.card.get_i(SPC_TRIG_DELAY)
    
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
    def ext0_level0(self, level : int = None) -> int:
        """
        Set the trigger level 0 for the ext0 trigger (see register 'SPC_TRIG_EXT0_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level 0 for the ext0 trigger in mV
        
        Returns
        -------
        int
            The trigger level 0 for the ext0 trigger in mV
        """

        if level is not None:
            self.card.set_i(SPC_TRIG_EXT0_LEVEL0, level)
        return self.card.get_i(SPC_TRIG_EXT0_LEVEL0)
    
    def ext0_level1(self, level : int = None) -> int:
        """
        Set the trigger level 1 for the ext0 trigger (see register 'SPC_TRIG_EXT0_LEVEL1' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level for the ext0 trigger in mV
        
        Returns
        -------
        int
            The trigger level for the ext0 trigger in mV
        """

        if level is not None:
            self.card.set_i(SPC_TRIG_EXT0_LEVEL1, level)
        return self.card.get_i(SPC_TRIG_EXT0_LEVEL1)
    
    def ext1_level0(self, level : int = None) -> int:
        """
        Set the trigger level 0 for the ext1 trigger (see register 'SPC_TRIG_EXT1_LEVEL0' in chapter `Trigger` in the manual)
        
        Parameters
        ----------
        level : int
            The trigger level 0 for the ext1 trigger in mV
        
        Returns
        -------
        int
            The trigger level 0 for the ext1 trigger in mV
        """

        if level is not None:
            self.card.set_i(SPC_TRIG_EXT1_LEVEL0, level)
        return self.card.get_i(SPC_TRIG_EXT1_LEVEL0)