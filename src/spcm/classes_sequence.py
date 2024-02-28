# -*- coding: utf-8 -*-

from .constants import *

from .classes_data_transfer import DataTransfer

class Sequence(DataTransfer):
    """
    a high-level class to control the sequence mode on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)

    def max_segments(self, max_segments : int = 0) -> int:
        """
        Set the maximum number of segments that can be used in the sequence mode (see register 'SPC_SEQMODE_MAXSEGMENTS' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        max_segments : int
            The maximum number of segments that can be used in the sequence mode

        Returns
        -------
        max_segments : int
            The actual maximum number of segments that can be used in the sequence mode
        """
        if max_segments: 
            self.card.set_i(SPC_SEQMODE_MAXSEGMENTS, max_segments)
        return self.card.get_i(SPC_SEQMODE_MAXSEGMENTS)
    
    def write_segment(self, segment : int = None) -> int:
        """
        Defines the current segment to be addressed by the user. Must be programmed prior to changing any segment parameters. (see register 'SPC_SEQMODE_WRITESEGMENT' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        segment : int
            The segment to be addresses

        Returns
        -------
        segment : int
            The segment to be addresses
        """

        if segment is not None:
            self.card.set_i(SPC_SEQMODE_WRITESEGMENT, segment)
        return self.card.get_i(SPC_SEQMODE_WRITESEGMENT)
    
    def segment_size(self, segment_size : int = None) -> int:
        """
        Defines the number of valid/to be replayed samples for the current selected memory segment in samples per channel. (see register 'SPC_SEQMODE_SEGMENTSIZE' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        segment_size : int
            The size of the segment in samples

        Returns
        -------
        segment_size : int
            The size of the segment in samples
        """

        if segment_size is not None:
            self.card.set_i(SPC_SEQMODE_SEGMENTSIZE, segment_size)
        return self.card.get_i(SPC_SEQMODE_SEGMENTSIZE)
    
    def step_memory(self, step_index : int, next_step_index : int = None, segment_index : int = None, loops : int = None, flags : int = None) -> tuple[int, int, int, int]:
        """
        Defines the step memory for the current selected memory segment. (see register 'SPC_SEQMODE_STEPMEM0' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        step_index : int
            The index of the current step
        next_step_index : int
            The index of the next step in the sequence
        segment_index : int
            The index of the segment associated to the step
        loops : int
            The number of times the segment is looped 
        flags : int
            The flags for the step

        Returns
        -------
        next_step_index : int
            The index of the next step in the sequence
        segment_index : int
            The index of the segment associated to the step
        loops : int
            The number of times the segment is looped 
        flags : int
            The flags for the step

        """
        qwSequenceEntry = 0

        # setup register value
        if next_step_index is not None and segment_index is not None and loops is not None and flags is not None:
            qwSequenceEntry = (flags & ~SPCSEQ_LOOPMASK) | (loops & SPCSEQ_LOOPMASK)
            qwSequenceEntry <<= 32
            qwSequenceEntry |= ((next_step_index << 16) & SPCSEQ_NEXTSTEPMASK) | (int(segment_index) & SPCSEQ_SEGMENTMASK)
            self.card.set_i(SPC_SEQMODE_STEPMEM0 + step_index, qwSequenceEntry)
        
        qwSequenceEntry = self.card.get_i(SPC_SEQMODE_STEPMEM0 + step_index)
        return (qwSequenceEntry & SPCSEQ_NEXTSTEPMASK) >> 16, qwSequenceEntry & SPCSEQ_SEGMENTMASK, (qwSequenceEntry >> 32) & SPCSEQ_LOOPMASK, (qwSequenceEntry >> 32) & ~SPCSEQ_LOOPMASK

    
    def start_step(self, start_step_index : int = None) -> int:
        """
        Defines which of all defined steps in the sequence memory will be used first directly after the card start. (see register 'SPC_SEQMODE_STARTSTEP' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        start_step_index : int
            The index of the start step

        Returns
        -------
        start_step_index : int
            The index of the start step
        """

        if start_step_index is not None:
            self.card.set_i(SPC_SEQMODE_STARTSTEP, start_step_index)
        return self.card.get_i(SPC_SEQMODE_STARTSTEP)
    
    def status(self) -> int:
        """
        Reads the status of the sequence mode. (see register 'SPC_SEQMODE_STATUS' in chapter `Sequence Mode` in the manual)

        Returns
        -------
        status : int
            The status of the sequence mode

        """
        return self.card.get_i(SPC_SEQMODE_STATUS)