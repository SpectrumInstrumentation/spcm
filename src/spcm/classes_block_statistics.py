# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from spcm_core.constants import *
from spcm_core import *

from . import units

from .classes_multi import Multi
from .classes_unit_conversion import UnitConversion
from .classes_error_exception import SpcmException

SPCM_SEGSTAT_STRUCT_CHx = np.dtype([
    ('average', 'i8'), # llAvrg
    ('minimum', 'i2'), # nMin
    ('maximum', 'i2'), # nMax
    ('minimum_position', 'i4'), # dwMinPos
    ('maximum_position', 'i4'), # dwMaxPos
    ('_unused', 'i4'), # _Unused
    ('timestamp', 'i8') # qw_Timestamp
])

class BlockStatistics(Multi):
    """a high-level class to control Block Statistics functionality on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
    
    def _bits_per_sample(self) -> int:
        """
        Get the number of bits per sample

        Returns
        -------
        int
            number of bits per sample
        """
        self.bits_per_sample = SPCM_SEGSTAT_STRUCT_CHx.itemsize * 8
        return self.bits_per_sample
    
    def _bytes_per_sample(self) -> int:
        """
        Get the number of bytes per sample

        Returns
        -------
        int
            number of bytes per sample
        """
        self.bytes_per_sample = SPCM_SEGSTAT_STRUCT_CHx.itemsize
        return self.bytes_per_sample

    def allocate_buffer(self, segment_samples : int, num_segments : int = None) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        segment_samples : int | pint.Quantity
            use the number of samples and get the number of active channels and bytes per samples directly from the card
        num_segments : int = None
            the number of segments that are used for the multiple recording mode
        """

        segment_samples = UnitConversion.convert(segment_samples, units.S, int)
        num_segments = UnitConversion.convert(num_segments, units.S, int)

        self.buffer_samples = num_segments # There is always just one sample (statistics block) per segment
        self._num_segments = num_segments
        self.segment_samples(segment_samples)
        
        num_channels = self.card.active_channels()
        self.buffer = np.empty((self._num_segments, num_channels), dtype=self.numpy_type())

    def numpy_type(self) -> npt.NDArray[np.int_]:
        """
        Get the type of numpy data from number of bytes

        Returns
        -------
        numpy data type
            the type of data that is used by the card
        """
        
        return SPCM_SEGSTAT_STRUCT_CHx
    