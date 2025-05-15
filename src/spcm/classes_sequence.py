# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from spcm_core.constants import *
from spcm_core import spcm_dwDefTransfer_i64, c_void_p

from .classes_card import Card
from .classes_data_transfer import DataTransfer

from .classes_unit_conversion import UnitConversion
from . import units


class Segment:
    """
    a class that contains information about a segment in the sequence mode
    (see register 'SPC_SEQMODE_WRITESEGMENT' in chapter `Sequence Mode` in the manual)
    """

    _card : Card = None
    _index : int = None
    _segment : npt.NDArray = None
    _samples_per_segment : int = 0
    def __init__(self, card : Card, index : int, segment : npt.NDArray, samples_per_segment : int) -> None:
        self._card = card
        self._index = index
        self._segment = segment
        self._samples_per_segment = samples_per_segment
    
    def __str__(self) -> str:
        return f"Segment {self._index}"
    
    def __int__(self) -> int:
        return self._index
    __index__ = __int__
    
    def __getitem__(self, item) -> npt.NDArray:
        """
        Returns the segment data for the given index

        Parameters
        ----------
        item : int | slice
            The index or slice of the segment data to be returned

        Returns
        -------
        segment : npt.NDArray
            The segment data for the given index
        """
        return self._segment[item]

    def __setitem__(self, item, value) -> None:
        """
        Sets the segment data for the given index

        Parameters
        ----------
        item : int | slice
            The index or slice of the segment data to be set
        value : npt.NDArray
            The value to be set for the given index
        """
        
        self._segment[item] = value
    
    def __getattr__(self, name):
        """
        Returns specific attributes of the segment array

        Parameters
        ----------
        name : str
            The name of the attribute to be returned    
        """
        
        return getattr(self._segment, name)

    def __len__(self) -> int:
        """
        Returns the length of the segment data

        Returns
        -------
        length : int
            The length of the segment data
        """
        
        return self._samples_per_segment
    
    def index(self) -> int:
        """
        Returns the index of the segment

        Returns
        -------
        index : int
            The index of the segment
        """
        
        return self._index
    
    def update(self) -> None:
        """
        Starts the buffer transfer for the given segment
        """

        self._card.set_i(SPC_SEQMODE_WRITESEGMENT, self.index())
        self._card.set_i(SPC_SEQMODE_SEGMENTSIZE, len(self))
        
        # we define the buffer for transfer
        self._card._print("Starting the DMA transfer and waiting until data is in board memory")
        _c_buffer = self.ctypes.data_as(c_void_p)
        self._card._check_error(spcm_dwDefTransfer_i64(self._card._handle, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, _c_buffer, 0, self.nbytes))

        self._card.cmd(M2CMD_DATA_STARTDMA, M2CMD_DATA_WAITDMA)


class Step:
    """
    a class that contains information about a step in the sequence mode
    (see register 'SPC_SEQMODE_STEPMEM0' in chapter `Sequence Mode` in the manual)
    """

    _card : Card = None
    _index : int = None
    _segment : Segment = None
    _loops : int = 0
    
    _to_step : "Step" = None
    _flags : bool = False

    def __init__(self, card : Card, index : int, segment : Segment, loops : int = 0) -> None:
        self._card = card
        self._index = index
        self._segment = segment
        self._loops = loops
        self._to_step = None
        self._flags = False
    
    def __str__(self) -> str:
        return f"Step {self._index} | {self._segment} | loops: {self._loops}"

    def __int__(self) -> int:
        return self._index
    __index__ = __int__

    def index(self) -> int:
        """
        Returns the index of the step

        Returns
        -------
        index : int
            The index of the step
        """

        return self._index
    
    def segment(self) -> Segment:
        """
        Returns the segment associated to the step

        Returns
        -------
        segment : Segment
            The segment associated to the step
        """

        return self._segment

    def set_transition(self, to_step : "Step", on_trig : bool = False):
        """
        Sets a transition between steps to the sequence mode

        Parameters
        ----------
        to_step : Step
            The step to which the transition is made
        on_trig : bool
            If True, the transition is made on trigger
        """
        
        self._to_step = to_step
        self._flags = SPCSEQ_ENDLOOPONTRIG if on_trig else SPCSEQ_ENDLOOPALWAYS
        # self.memory(self._to_step, SPCSEQ_ENDLOOPONTRIG if on_trig else SPCSEQ_ENDLOOPALWAYS)
    
    def final_step(self):
        """
        Sets the step as final step in the sequence mode
        """
        
        self._to_step = None
        self._flags = SPCSEQ_END
        # self.memory(None, SPCSEQ_END)
    final = final_step
    
    # def update(self):
    #     """
    #     Updates the step with the given parameters
    #     """

    #     self.transfer()
    
    def update(self, segment : Segment = None, loops : int = None, to_step : "Step" = None, on_trig : bool = None) -> None:
        """
        Defines or updates the step memory for the current selected memory segment. (see register 'SPC_SEQMODE_STEPMEM0' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        segment : Segment
            The segment associated to the step
        loops : int
            The number of times the segment is looped
        to_step : Step
            The step to which the transition is made
        on_trig : bool
            If True, the transition is made on trigger
        """

        if segment is not None:
            self._segment = segment
        if loops is not None:
            self._loops = loops
        if to_step is not None:
            self._to_step = to_step
        if on_trig is not None:
            self._flags = SPCSEQ_ENDLOOPONTRIG if on_trig else SPCSEQ_ENDLOOPALWAYS

        entry = 0
        next_step_index = 0
        if isinstance(self._to_step, Step):
            next_step_index = self._to_step.index()
        
        entry = (self._flags & ~SPCSEQ_LOOPMASK) | (self._loops & SPCSEQ_LOOPMASK)
        entry <<= 32
        entry |= ((next_step_index << 16) & SPCSEQ_NEXTSTEPMASK) | (self._segment.index() & SPCSEQ_SEGMENTMASK)
        self._card.set_i(SPC_SEQMODE_STEPMEM0 + self.index(), entry)


class Sequence(DataTransfer):
    """
    a high-level class to control the sequence mode on Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    segments : list[Segment] = []
    steps : list[Step] = []
    _entry_step : Step = None
    # _final_step : Step = None

    def __init__(self, card, *args, **kwargs) -> None:
        super().__init__(card, *args, **kwargs)
        self.segments = []
        self.steps = []
        self.transitions = {}
        self._entry_step = None
        # self._final_step = None

    def __str__(self) -> str:
        return_string = ""
        for i in range(len(self.steps)):
            next_step_index, seqment_index, loops, flags = self.step_memory(i)
            return_string += f"Step {i}: next {next_step_index}, segment {seqment_index}, loops {loops}, flags 0b{flags:b}\n"
        return return_string
    __repr__ = __str__

    ### Low-level sequence control ###
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
    
    def write_segment(self, segment = None) -> int:
        """
        Defines the current segment to be addressed by the user. Must be programmed prior to changing any segment parameters. (see register 'SPC_SEQMODE_WRITESEGMENT' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        segment : int | Segment
            The segment to be addresses

        Returns
        -------
        segment : int
            The segment to be addresses
        """

        if segment is not None:
            if isinstance(segment, Segment):
                segment = int(segment)
            self.card.set_i(SPC_SEQMODE_WRITESEGMENT, segment)
        return self.card.get_i(SPC_SEQMODE_WRITESEGMENT)
    
    def segment_size(self, segment_size : int = None, return_unit = None) -> int:
        """
        Defines the number of valid/to be replayed samples for the current selected memory segment in samples per channel. (see register 'SPC_SEQMODE_SEGMENTSIZE' in chapter `Sequence Mode` in the manual)

        Parameters
        ----------
        segment_size : int | pint.Quantity
            The size of the segment in samples

        Returns
        -------
        segment_size : int
            The size of the segment in samples
        """

        if segment_size is not None:
            segment_size = UnitConversion.convert(segment_size, units.Sa, int)
            self.card.set_i(SPC_SEQMODE_SEGMENTSIZE, segment_size)
        return_value = self.card.get_i(SPC_SEQMODE_SEGMENTSIZE)
        if return_unit is not None: return UnitConversion.to_unit(return_value, return_unit)
        return return_value
    
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
    
    def current_step(self) -> Step:
        """
        Returns the current step of the sequence mode

        Returns
        -------
        step : Step
            The current step of the sequence mode
        """

        current_step_index = self.status()
        return self.steps[current_step_index] if current_step_index < len(self.steps) else None
    
    ### High-level Step and Segment handling ###
    def add_segment(self, length : int = None) -> Segment:
        """
        Adds a segment to the sequence mode

        Returns
        -------
        segment : Segment
            The segment that was added
        """

        index = len(self.segments)
        segment_array = self._allocate_buffer(length, False, self.num_channels)
        segment = Segment(self.card, index, segment_array, length)
        self.segments.append(segment)
        return segment
    
    def add_step(self, segment : Segment, loops : int = 0) -> Step:
        """
        Adds a step to the sequence mode

        Parameters
        ----------
        segment : Segment
            The segment associated to the step
        loops : int
            The number of times the segment is looped

        Returns
        -------
        step : Step
            The step that was added
        """

        index = len(self.steps)
        step = Step(self.card, index, segment = segment, loops = loops)
        self.steps.append(step)
        return step
    
    def entry_step(self, step : Step = None) -> Step:
        """
        Returns and sets the entry point of the step

        Parameters
        ----------
        step : Step
            The step that will be set as entry point

        Returns
        -------
        Step
            The step that is used a entry point
        """

        index = None
        if step is not None:
            self._entry_step = step
            index = step.index()
        index = self.start_step(index)
        return self.steps[index]
    
    def write_setup(self):
        """
        Writes the setup to the card
        """
        
        num_segments = len(self.segments)
        num_segments_pow2 = np.power(2, np.ceil(np.log2(num_segments))).astype(np.int64)
        self.max_segments(num_segments_pow2)

        # write segments
        for segment in self.segments:
            segment.update()

        # write steps
        for step in self.steps:
            step.update()
        
        self.card._print("Finished writing the sequence data to the card")
        
    def transfer_segment(self, segment : Segment) -> None:
        """
        Starts the buffer transfer for the given segment
        """

        segment.update()

        # self.write_segment(segment)
        # self.segment_size(len(segment))

        # transfer_offset_bytes = 0
        # transfer_length_bytes = segment.nbytes

        # buffer_type = SPCM_BUF_DATA
        # direction = SPCM_DIR_PCTOCARD
        
        # # we define the buffer for transfer
        # self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        # _c_buffer = segment.ctypes.data_as(c_void_p)
        # self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, buffer_type, direction, self.notify_size, _c_buffer, transfer_offset_bytes, transfer_length_bytes))

        # self.card.cmd(M2CMD_DATA_STARTDMA, M2CMD_DATA_WAITDMA)
        
    
