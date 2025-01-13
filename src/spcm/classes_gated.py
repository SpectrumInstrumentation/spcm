# -*- coding: utf-8 -*-
import numpy as np

from spcm_core.constants import *

from .classes_data_transfer import DataTransfer
from .classes_time_stamp import TimeStamp

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

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
        if self.direction is Direction.Acquisition:
            self.max_num_gates = kwargs.get("max_num_gates", 128)
            self.timestamp = TimeStamp(card)
            self.timestamp.mode(SPC_TSMODE_STARTRESET, SPC_TSCNT_INTERNAL)
            self.timestamp.allocate_buffer(2*self.max_num_gates)
            self._alignment = self.card.get_i(SPC_GATE_LEN_ALIGNMENT)
    
    def start_buffer_transfer(self, *args, **kwargs):
        super().start_buffer_transfer(*args, **kwargs)
        if self.direction is Direction.Acquisition:
            self.timestamp.start_buffer_transfer(M2CMD_EXTRA_STARTDMA)
    
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

        if self._memory_size < num_samples:
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
    
    def gate_counter(self) -> int:
        """
        Get the number of gate events since acquisition start (see register 'SPC_TRIGGERCOUNTER' in chapter `Trigger` in the manual)
        
        Returns
        -------
        int
            The gate counter
        """

        return self.card.get_i(SPC_TRIGGERCOUNTER)
    
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
    def __next__(self):
        if self.direction is not Direction.Acquisition:
            raise ValueError("Iterating the Gated class can only be used with acquisition")
        self.iterator_index += 1
        self.gate_count = self.gate_counter()
        if self.iterator_index >= self.gate_count:
            self.iterator_index = -1
            raise StopIteration
        # Get the start and end of the gate event
        alignment = self.alignment()
        length_unaligned = self.timestamp.buffer[2*self.iterator_index+1, 0] - self.timestamp.buffer[2*self.iterator_index+0, 0]
        length_aligned = (length_unaligned // alignment + 1) * alignment
        segment_length = length_unaligned + self._pre_trigger + self._post_trigger
        total_length = length_aligned + self._pre_trigger + self._post_trigger
        end = self._start + total_length
        self._end = self._start + segment_length
        if end > self.buffer.size:
            print("Warning: Gate exceeds data length")
            total_length -= end - self.buffer.size
            segment_length -= self._end - self.buffer.size
            end = self.buffer.size
            self._end = self.buffer.size
        return self.buffer[:, self._start:self._end]
    
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

        time_range = np.arange(self.timestamp.buffer[2*self.iterator_index+0, 0] - self._pre_trigger, self.timestamp.buffer[2*self.iterator_index+1, 0] + self._post_trigger)
        time_range = UnitConversion.to_unit(time_range / self._sample_rate(), return_unit)
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