# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt
import time

from spcm_core.constants import *

from .classes_data_transfer import DataTransfer
from .classes_time_stamp import TimeStamp
from .classes_error_exception import SpcmException

from .classes_unit_conversion import UnitConversion
from . import units


class Gated(DataTransfer):
    """
    A high-level class to control Gated sampling Spectrum Instrumentation cards.

    For more information about what setups are available, please have a look at the user manual
    for your specific card.
    """

    max_num_gates : int = 128
    timestamp : TimeStamp = None

    # Private
    _pre_trigger : int = 0
    _post_trigger : int = 0

    _timestamp : TimeStamp = None
    _alignment : int = 0

    # private FIFO mode settings
    _fifo_mode : bool = False
    _num_gates : int = 0

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
        if self.direction is Direction.Acquisition:
            self.max_num_gates = kwargs.get("max_num_gates", 128)
            if card.card_mode() == SPC_REC_FIFO_GATE:
                print("Gated acquisition mode is set to FIFO mode")
                self._fifo_mode = True
                self.num_gates(kwargs.get("num_gates", 0))
                if self._num_gates > 0:
                    self.max_num_gates = self._num_gates
            self.timestamp = TimeStamp(card)
            self.timestamp.mode(SPC_TSMODE_STARTRESET, SPC_TSCNT_INTERNAL)
            self.timestamp.allocate_buffer(2*self.max_num_gates)
            self._alignment = self.card.get_i(SPC_GATE_LEN_ALIGNMENT)
    
    def start_buffer_transfer(self, *args, **kwargs):
        super().start_buffer_transfer(*args, **kwargs)
        if self.direction is Direction.Acquisition:
            # self.timestamp.start_buffer_transfer(M2CMD_EXTRA_STARTDMA)
            self.timestamp.start_buffer_transfer(M2CMD_EXTRA_POLL)
    
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

        if not self._fifo_mode and self._memory_size < num_samples:
            raise ValueError("The number of post trigger samples needs to be smaller than the total number of samples")
        if num_samples is not None:
            num_samples = UnitConversion.convert(num_samples, units.Sa, int)
            self.card.set_i(SPC_POSTTRIGGER, num_samples)
        self._post_trigger = self.card.get_i(SPC_POSTTRIGGER)
        return self._post_trigger

    def alignment(self) -> int:
        """
        Get the alignment of the end of the gated data buffer (see register `SPC_GATE_LEN_ALIGNMENT` in the manual)

        Returns
        -------
        int
            the number of samples to align the end of the gated data buffer
        """
        return self._alignment
    
    def num_gates(self, num_gates : int = None) -> int:
        """
        FIFO only: set the number of gates to be acquired (see register `SPC_LOOPS` in the manual)

        Parameters
        ----------
        num_gates : int
            the number of gates to be acquired

        Returns
        -------
        int
            the number of gates to be acquired
        """
        
        if num_gates is not None:
            self.card.set_i(SPC_LOOPS, num_gates)
        self._num_gates = self.card.get_i(SPC_LOOPS)
        return self._num_gates
    
    def gate_counter(self) -> int:
        """
        Get the number of gate events since acquisition start (see register 'SPC_TRIGGERCOUNTER' in chapter `Trigger` in the manual)
        
        Returns
        -------
        int
            The gate counter
        """

        return self.card.get_i(SPC_TRIGGERCOUNTER)
    
    def available_gates(self) -> tuple[int, int]:
        """
        Get the number of available gates in the timestamp buffer

        Returns
        -------
        tuple(int, int)
            The current position and the number of available gates from the timestamp buffer
        """
        
        return self.timestamp.avail_user_pos() // 2, self.timestamp.avail_user_len() // 2
    
    def __iter__(self):
        """
        Returns an iterator object and initializes the iterator index.

        Returns
        -------
        iterable
            An iterator object for the class.
        """
        iter = super().__iter__()
        self.iterator_index = -1
        return iter
    
    iterator_index : int = -1
    gate_count : int = 0
    _start : int = 0
    _end : int = 0
    _aligned_end : int = 0
    _current_num_samples : int = 0
    def __next__(self):
        if self.direction is not Direction.Acquisition:
            raise ValueError("Iterating the Gated class can only be used with acquisition")
        if self._fifo_mode and not self._polling:
            raise SpcmException("Polling is required for fifo gated acquisition. Please set the polling mode to True")

        self.iterator_index += 1

        # notify the card that data is available or read, but only after the first block
        if self.iterator_index > 0 and self._auto_avail_card_len:
            self.flush()

        # Check if all the gates have been acquired
        if self._num_gates > 0 and self.iterator_index >= self._num_gates: self.stop_next()
        ts_len = self.timestamp.avail_user_len()
        if not self._fifo_mode and ts_len < 2: self.stop_next()
        
        while self._polling:
            ts_len = self.timestamp.avail_user_len()
            self.card._print(f"Available time stamps: {ts_len}", end="\r")
            if ts_len >= 2:
                break
            time.sleep(self._polling_timer)

        # Get the start and end of the gate event
        self._start = self.avail_user_pos()
        length = self.timestamp.buffer[2*self.iterator_index+1, 0] - self.timestamp.buffer[2*self.iterator_index+0, 0]
        self.card._print(f"Gate {self.iterator_index} - Start: {self._start} - Length: {length}", end="\n")
        segment_length = length + self._pre_trigger + self._post_trigger
        self._end = self._start + segment_length

        # The data end of the gate is aligned to a multiple of the alignment length, hence we have to calculate the aligned end of the gate to know where the next gate starts
        alignment = self.alignment()
        length_with_alignment = (length // alignment + 1) * alignment
        self._current_num_samples = length_with_alignment + self._pre_trigger + self._post_trigger
        self._aligned_end = self._start + self._current_num_samples

        # Wait for enough data to be available in the buffer to get the next gate
        while self._polling:
            user_len = self.avail_user_len()
            self.card._print(f"Available data: {user_len} - Required data: {self._current_num_samples}", end="\r")
            if user_len >= self._current_num_samples:
                break
            time.sleep(self._polling_timer)

        self._current_samples += self._current_num_samples
        if self._to_transfer_samples > 0 and self._to_transfer_samples <= self._current_samples:
            self.stop_next()
        
        # Return the view of the data buffer that contains only the data of the current gate
        return self.buffer[:, self._start:self._end]
    
    def stop_next(self):
        """
        Stop the iteration and flush all the iterator parameters
        """
        self.iterator_index = -1
        self._start = 0
        self._end = 0
        self._aligned_end = 0
        self._current_samples = 0
        self._current_num_samples = 0
        raise StopIteration
    
    def flush(self):
        """
        This method is used to tell the card that a notify size of data is freed up after reading (acquisition) or written to (generation)
        """
        self.avail_card_len(self._current_num_samples)
        self.timestamp.avail_card_len(2) # two time stamps per gate
    
    def current_time_range(self, return_unit = None) -> int:
        """
        Get the current time range of the data buffer

        Parameters
        ----------
        return_unit : pint.Unit
            the unit to return the time range in

        Returns
        -------
        int or pint.Quantity
            the current time range of the data buffer
        """

        current_length = self._end - self._start
        time_range = np.arange(self.timestamp.buffer[2*self.iterator_index+0, 0] - self._pre_trigger, self.timestamp.buffer[2*self.iterator_index+1, 0] + self._post_trigger)
        time_range = UnitConversion.to_unit(time_range[:current_length] / self._sample_rate(), return_unit)
        return time_range

    def current_timestamps(self, return_unit = None) -> tuple:
        """
        Get the current timestamps of the data buffer

        Parameters
        ----------
        return_unit : pint.Unit
            the unit to return the timestamps in

        Returns
        -------
        tuple
            the current timestamps of the data buffer
        """

        ts_start = self.timestamp.buffer[2*self.iterator_index+0, 0]
        ts_end = self.timestamp.buffer[2*self.iterator_index+1, 0]
        ts_start = UnitConversion.to_unit(ts_start / self._sample_rate(), return_unit)
        ts_end = UnitConversion.to_unit(ts_end / self._sample_rate(), return_unit)
        return ts_start, ts_end