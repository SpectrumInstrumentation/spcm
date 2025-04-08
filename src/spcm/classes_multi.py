# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt

from spcm_core.constants import *

from .classes_data_transfer import DataTransfer

import pint
from .classes_unit_conversion import UnitConversion
from . import units

class Multi(DataTransfer):
    """a high-level class to control Multiple Recording and Replay functionality on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    # Private
    _segment_size : int
    _num_segments : int

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
        self._pre_trigger = None
        self._segment_size = 0
        self._num_segments = 0

    def segment_samples(self, segment_size : int = None) -> None:
        """
        Sets the memory size in samples per channel. The memory size setting must be set before transferring 
        data to the card. (see register `SPC_MEMSIZE` in the manual)
        
        Parameters
        ----------
        segment_size : int | pint.Quantity
            the size of a single segment in memory in Samples
        """

        if segment_size is not None:
            segment_size = UnitConversion.convert(segment_size, units.S, int)
            self.card.set_i(SPC_SEGMENTSIZE, segment_size)
        segment_size = self.card.get_i(SPC_SEGMENTSIZE)
        self._segment_size = segment_size
    
    def post_trigger(self, num_samples : int = None) -> int:
        """
        Set the number of post trigger samples (see register `SPC_POSTTRIGGER` in the manual)
        
        Parameters
        ----------
        num_samples : int | pint.Quantity
            the number of post trigger samples
        
        Returns
        -------
        int
            the number of post trigger samples
        """

        post_trigger = super().post_trigger(num_samples)
        self._pre_trigger = self._segment_size - post_trigger
        return post_trigger

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
        self.segment_samples(segment_samples)
        if num_segments is None:
            self._num_segments = self._memory_size // segment_samples
        else:
            self._num_segments = num_segments
        
        super().allocate_buffer(segment_samples * self._num_segments, no_reshape=True)

        num_channels = self.card.active_channels()
        if self.bits_per_sample > 1 and not self._12bit_mode:
            self.card._print(f"{self._num_segments} segments of {segment_samples} samples with {num_channels} channels")
            self.buffer = self.buffer.reshape((self._num_segments, segment_samples, num_channels), order='C') # index definition: [segment, sample, channel] !

    def time_data(self, total_num_samples : int = None, return_units = units.s) -> npt.NDArray:
        """
        Get the time array for the data buffer

        Parameters
        ----------
        total_num_samples : int | pint.Quantity
            the total number of samples
        return_units : pint.Unit
            the units of the time array
        
        Returns
        -------
        numpy array
            the time array
        """

        if total_num_samples is None:
            total_num_samples = self._buffer_samples // self._num_segments
        return super().time_data(total_num_samples, return_units)

    def unpack_12bit_buffer(self, data : npt.NDArray[np.int_] = None) -> npt.NDArray[np.int_]:
        """
        Unpacks the 12-bit packed data to 16-bit data

        Parameters
        ----------
        data : numpy array
            the packed data

        Returns
        -------
        numpy array
            the unpacked 16bit buffer
        """
        buffer_12bit = super().unpack_12bit_buffer(data)
        return buffer_12bit.reshape((self._num_segments, self.num_channels, self._segment_size), order='C')

    
    def __next__(self) -> npt.ArrayLike:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        npt.ArrayLike
            the next data block
        
        Raises
        ------
        StopIteration
        """
        super().__next__()
        user_pos = self.avail_user_pos()
        current_segment = user_pos // self._segment_size
        current_pos_in_segment = user_pos % self._segment_size
        final_segment = ((user_pos+self._notify_samples) // self._segment_size)
        final_pos_in_segment = (user_pos+self._notify_samples) % self._segment_size

        self.card._print("NumSamples = {}, CurrentSegment = {}, CurrentPos = {},  FinalSegment = {}, FinalPos = {}".format(self._notify_samples, current_segment, current_pos_in_segment, final_segment, final_pos_in_segment))

        return self.buffer[current_segment:final_segment, :, :]
