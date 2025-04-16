# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt
import pint

from spcm_core.constants import *

from .classes_data_transfer import DataTransfer

class ABA(DataTransfer):
    """a high-level class to control ABA functionality on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    # Private
    _divider : int = 1

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
        self.buffer_type = SPCM_BUF_ABA
        self._divider = 1
    
    def divider(self, divider : int = None) -> int:
        """
        Set the divider for the ABA functionality (see register `SPC_ABADIVIDER` in the manual)

        Parameters
        ----------
        divider : int
            The divider for the ABA functionality

        Returns
        -------
        int
            The divider for the ABA functionality

        """
        
        if divider is not None:
            self.card.set_i(SPC_ABADIVIDER, divider)
        self._divider = self.card.get_i(SPC_ABADIVIDER)
        return self._divider
    
    def _sample_rate(self) -> pint.Quantity:
        """
        Get the sample rate of the ABA data of the card

        Returns
        -------
        pint.Quantity
            the sample rate of the card in Hz as a pint quantity
        """
        
        sample_rate = super()._sample_rate()
        return sample_rate / self._divider

    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_ABA, **kwargs) -> int:
        return super().start_buffer_transfer(*args, buffer_type=buffer_type, **kwargs)
