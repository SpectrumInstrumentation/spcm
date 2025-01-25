# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_dds_command_list import DDSCommandList

class DDSCommandQueue(DDSCommandList):
    """
    Abstraction class of the set_ptr and register `SPC_REGISTER_LIST` for command streaming in the DDS functionality.
    This class is used to write commands to the card in a more efficient way using a queuing mechanism that writes
    to the card when the queue is filled-up.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = self.WRITE_MODE.NO_CHECK

    def set_i(self, reg : int, value : int) -> None:
        """
        set an integer value to a register

        Parameters
        ----------
        reg : int
            the register to be changed
        value : int
            the value to be set
        """
        
        self.command_list[self.current_index].lReg = reg
        self.command_list[self.current_index].lType = TYPE_INT64
        self.command_list[self.current_index].llValue = value
        self.current_index += 1

        if self.current_index >= self.list_size:
            self.write_to_card()
    
    def set_d(self, reg : int, value) -> None:
        """
        set a double value to a register

        Parameters
        ----------
        reg : int
            the register to be changed
        value : np._float64
            the value to be set
        """
        
        self.command_list[self.current_index].lReg = reg
        self.command_list[self.current_index].lType = TYPE_DOUBLE
        self.command_list[self.current_index].dValue = value
        self.current_index += 1

        if self.current_index >= self.list_size:
            self.write_to_card()

    def write_to_card(self) -> None:
        """
        write the current list of commands to the card
        """

        super().write_to_card()
        self.write()

    def write(self) -> None:
        """
        write the current list of commands to the card and reset the current command index
        """

        super().write()
        self.current_index = 0

    # DDS "static" parameters
    def amp(self, index : int, amplitude : float) -> None:
        """
        set the amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP` in the manual)
        
        Parameters
        ----------
        index : int
            the core index
        amplitude : float
            the value between 0 and 1 corresponding to the amplitude
        """

        self.set_d(SPC_DDS_CORE0_AMP + index, amplitude)

    def freq(self, index : int, frequency : float) -> None:
        """
        set the frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ` in the manual)
        
        Parameters
        ----------
        index : int
            the core index
        frequency : float
            the value of the frequency in Hz
        """

        self.set_d(SPC_DDS_CORE0_FREQ + index, frequency)
    
    def phase(self, index : int, phase : float) -> None:
        """
        set the phase of the sine wave of a specific core (see register `SPC_DDS_CORE0_PHASE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        phase : float
            the value between 0 and 360 degrees of the phase
        """

        self.set_d(SPC_DDS_CORE0_PHASE + index, phase)
    
    def freq_slope(self, core_index : int, slope : float) -> None:
        """
        set the frequency slope of the linearly changing frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        slope : float
            the rate of frequency change in Hz/s
        """

        self.set_d(SPC_DDS_CORE0_FREQ_SLOPE + core_index, slope)

    def amp_slope(self, core_index : int, slope : float) -> None:
        """
        set the amplitude slope of the linearly changing amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        slope : float
            the rate of amplitude change in 1/s
        """

        self.set_d(SPC_DDS_CORE0_AMP_SLOPE + core_index, slope)

    def trg_timer(self, period : float) -> None:
        """
        set the period at which the timer should raise DDS trigger events. (see register `SPC_DDS_TRG_TIMER` in the manual)

        NOTE
        ----
        only used in conjecture with the trigger source set to SPCM_DDS_TRG_SRC_TIMER ---

        Parameters
        ----------
        period : float
            the time between DDS trigger events in seconds
        """

        self.set_d(SPC_DDS_TRG_TIMER, float(period))
