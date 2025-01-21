# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from spcm_core.constants import *

from .classes_multi import Multi

class Boxcar(Multi):
    """a high-level class to control Boxcar functionality on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
    
    def box_averages(self, num_averages : int = None) -> int:
        """Sets the number of averages for the boxcar functionality (see hardware reference manual register 'SPC_BOX_AVERAGES')

        Parameters
        ----------
        num_averages : int
            the number of averages for the boxcar functionality

        Returns
        -------
        int
            the number of averages for the boxcar functionality
        """
        if num_averages is not None:
            self.card.set_i(SPC_BOX_AVERAGES, num_averages)
        return self.card.get_i(SPC_BOX_AVERAGES)
    
    def _bits_per_sample(self) -> int:
        """
        Get the number of bits per sample

        Returns
        -------
        int
            number of bits per sample
        """
        self.bits_per_sample = 32
        return self.bits_per_sample
    
    def _bytes_per_sample(self) -> int:
        """
        Get the number of bytes per sample

        Returns
        -------
        int
            number of bytes per sample
        """
        self.bytes_per_sample = 4
        return self.bytes_per_sample

    def numpy_type(self) -> npt.NDArray[np.int_]:
        """
        Get the type of numpy data from number of bytes

        Returns
        -------
        numpy data type
            the type of data that is used by the card
        """
        return np.int32
    