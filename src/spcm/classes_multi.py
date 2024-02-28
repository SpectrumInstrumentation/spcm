# -*- coding: utf-8 -*-

import numpy.typing as npt

from .constants import *

from .classes_data_transfer import DataTransfer

from .classes_error_exception import SpcmTimeout

class Multi(DataTransfer):
    """a high-level class to control Multiple Recording and Replay functionality on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    # Private
    _segment_size : int = 0

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)

    def segment_samples(self, segment_size : int) -> None:
        """
        Sets the memory size in samples per channel. The memory size setting must be set before transferring 
        data to the card. (see register `SPC_MEMSIZE` in the manual)
        
        Parameters
        ----------
        segment_size : int
            the size of a single segment in memory in Bytes
        """
        self.card.set_i(SPC_SEGMENTSIZE, segment_size)
        self._segment_size = segment_size
    
    def allocate_buffer(self, segment_samples : int, num_segments : int) -> None:
        """Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples_per_segment : int = None
            use the number of samples and get the number of active channels and bytes per samples directly from the card
        num_segments : int
            the number of segments that are used for the multiple recording mode
        """
        
        self.segment_samples(segment_samples)
        super().allocate_buffer(segment_samples * num_segments)
        num_channels = self.card.active_channels()
        if self.bits_per_sample > 1:
            self.buffer = self.buffer.reshape((num_channels, num_segments, segment_samples), order='C')
    
    
    
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
        timeout_counter = 0
        # notify the card that data is available or read, but only after the first block
        if self._current_samples >= self._notify_samples:
            self.avail_card_len(self._notify_samples)
        while True:
            try:
                self.wait_dma()
            except SpcmTimeout:
                self.card._print("... Timeout ({})".format(timeout_counter), end='\r')
                timeout_counter += 1
                if timeout_counter > self._max_timeout:
                    raise StopIteration
            else:
                user_len = self.avail_user_len()
                user_pos = self.avail_user_pos()

                current_segment = user_pos // self._segment_size
                current_pos_in_segment = user_pos % self._segment_size
                final_segment = (user_pos+user_len) // self._segment_size
                final_pos_in_segment = (user_pos+user_len) % self._segment_size

                print("NumSamples = {}, CurrentSegment = {}, CurrentPos = {},  FinalSegment = {}, FinalPos = {}, UserLen = {}".format(self._notify_samples, current_segment, current_pos_in_segment, final_segment, final_pos_in_segment, user_len))

                self._current_samples += self._notify_samples
                if self._to_transfer_samples != 0 and self._to_transfer_samples < self._current_samples:
                    raise StopIteration

                fill_size = self.fill_size_promille()
                self.card._print("Fill size: {}%  Pos:{:08x} Len:{:08x} Total:{:.2f} MiS / {:.2f} MiS".format(fill_size/10, user_pos, user_len, self._current_samples / MEBI(1), self._to_transfer_samples / MEBI(1)), end='\r')

                return self.buffer[:, current_segment:final_segment, :]

    