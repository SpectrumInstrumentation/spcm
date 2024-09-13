# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_device import Device

class Sync(Device):
    """a class to control Spectrum Instrumentation Starhub synchronization devices

    For more information about what setups are available, please have a look at the user manual
    for your specific Starhub.
    
    Exceptions
    ----------
    SpcmException
    SpcmTimeout
    """

    def enable(self, enable : int = None) -> int:
        """
        Enable or disable the Starthub (see register 'SPC_SYNC_ENABLEMASK' in chapter `Star-Hub` in the manual)

        Parameters
        ----------
        enable : int or bool
            enable or disable the Starthub
        """

        if enable is not None:
            enable_mask = 0
            if isinstance(enable, bool):
                num_cards = self.sync_count()
                enable_mask = ((1 << num_cards) - 1) if enable else 0
            elif isinstance(enable, int):
                enable_mask = enable
            else:
                raise ValueError("The enable parameter must be a boolean or an integer")
            self.set_i(SPC_SYNC_ENABLEMASK, enable_mask)
        return self.get_i(SPC_SYNC_ENABLEMASK)
    
    def num_connectors(self) -> int:
        """
        Number of connectors that the Star-Hub offers at max. (see register 'SPC_SYNC_READ_NUMCONNECTORS' in chapter `Star-Hub` in the manual)

        Returns
        -------
        int
            number of connectors on the StarHub
        """

        return self.get_i(SPC_SYNC_READ_NUMCONNECTORS)
    
    def sync_count(self) -> int:
        """
        Number of cards that are connected to this Star-Hub (see register 'SPC_SYNC_READ_SYNCCOUNT' in chapter `Star-Hub` in the manual)

        Returns
        -------
        int
            number of synchronized cards
        """

        return self.get_i(SPC_SYNC_READ_SYNCCOUNT)
    
    def card_index(self, index) -> int:
        """
        Index of the card that is connected to the Star-Hub at the given local index (see register 'SPC_SYNC_READ_CARDIDX0' in chapter `Star-Hub` in the manual)

        Parameters
        ----------
        index : int
            connector index

        Returns
        -------
        int
            card index
        """

        return self.get_i(SPC_SYNC_READ_CARDIDX0 + index)
    
    def cable_connection(self, index) -> int:
        """
        Returns the index of the cable connection that is used for the logical connection `index`. (see register 'SPC_SYNC_READ_CABLECON0' in chapter `Star-Hub` in the manual)
        The cable connections can be seen printed on the PCB of the star-hub. Use these cable connection information in case 
        that there are hardware failures with the star-hub cabeling.

        Parameters
        ----------
        index : int
            connector index

        Returns
        -------
        int
            cable index
        """

        return self.get_i(SPC_SYNC_READ_CABLECON0 + index)