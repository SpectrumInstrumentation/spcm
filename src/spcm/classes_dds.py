# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_error_exception import SpcmException
from .classes_functionality import CardFunctionality
from .classes_channels import Channels, Channel

from .classes_unit_conversion import UnitConversion
from . import units

class DDSCore:
    """
    a class for controlling a single DDS core
    """

    dds : "DDS"
    index : int
    channel : Channel

    def __init__(self, core_index, dds, *args, **kwargs) -> None:
        self.dds = dds
        self.index = core_index
        self.channel = kwargs.get("channel", None)
    
    def __int__(self) -> int:
        """
        get the index of the core

        Returns
        -------
        int
            the index of the core
        """
        return self.index
    __index__ = __int__

    def __str__(self) -> str:
        """
        get the string representation of the core

        Returns
        -------
        str
            the string representation of the core
        """
        return f"Core {self.index}"
    __repr__ = __str__

    def __add__(self, other) -> int:
        """
        add the index of the core to another index

        Parameters
        ----------
        other : int
            the other index

        Returns
        -------
        int
            the sum of the two indices
        """
        return self.index + other

    # DDS "static" parameters
    def amp(self, amplitude : float) -> None:
        """
        set the amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP` in the manual)
        
        Parameters
        ----------
        amplitude : float | pint.Quantity
            the value between 0 and 1 corresponding to the amplitude
        """

        if self.channel is not None:
            amplitude = self.channel.to_amplitude_fraction(amplitude)
        elif isinstance(amplitude, units.Quantity) and amplitude.check("[]"):
            amplitude = UnitConversion.convert(amplitude, units.fraction, float, rounding=None)
        self.dds.set_d(SPC_DDS_CORE0_AMP + self.index, float(amplitude))
    # aliases
    amplitude = amp

    def get_amp(self, return_unit = None) -> float:
        """
        gets the amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP` in the manual)

        Parameters
        ----------
        return_unit : pint.Unit = None
            the unit of the returned amplitude, by default None

        Returns
        -------
        float
            the value between 0 and 1 corresponding to the amplitude
        """

        return_value = self.dds.card.get_d(SPC_DDS_CORE0_AMP + self.index)
        if self.channel is not None:
            return_value = self.channel.from_amplitude_fraction(return_value, return_unit)
        else:
            return_value = UnitConversion.to_unit(return_value, return_unit)
        return return_value
    # aliases
    get_amplitude = get_amp

    def freq(self, frequency : float) -> None:
        """
        set the frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ` in the manual)
        
        Parameters
        ----------
        frequency : float | pint.Quantity
            the value of the frequency in Hz
        """

        frequency = UnitConversion.convert(frequency, units.Hz, float, rounding=None)
        self.dds.set_d(SPC_DDS_CORE0_FREQ + self.index, float(frequency))
    # aliases
    frequency = freq

    def get_freq(self, return_unit = None) -> float:
        """
        gets the frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ` in the manual)
        
        Parameters
        ----------
        return_unit : pint.Unit = None
            the unit of the returned frequency, by default None

        Returns
        -------
        float | pint.Quantity
            the value of the frequency in Hz the specific core
        """

        return_value = self.dds.card.get_d(SPC_DDS_CORE0_FREQ + self.index)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.Hz, return_unit)
        return return_value
    # aliases
    get_frequency = get_freq

    def phase(self, phase : float) -> None:
        """
        set the phase of the sine wave of a specific core (see register `SPC_DDS_CORE0_PHASE` in the manual)
        
        Parameters
        ----------
        phase : float | pint.Quantity
            the value between 0 and 360 degrees of the phase
        """

        phase = UnitConversion.convert(phase, units.deg, float, rounding=None)
        self.dds.set_d(SPC_DDS_CORE0_PHASE + self.index, float(phase))

    def get_phase(self, return_unit = None) -> float:
        """
        gets the phase of the sine wave of a specific core (see register `SPC_DDS_CORE0_PHASE` in the manual)
        
        Returns
        -------
        float
            the value between 0 and 360 degrees of the phase
        """

        return_value = self.dds.card.get_d(SPC_DDS_CORE0_PHASE + self.index)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.deg, return_unit)
        return return_value

    # DDS dynamic parameters
    def freq_slope(self, slope : float) -> None:
        """
        set the frequency slope of the linearly changing frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ_SLOPE` in the manual)
        
        Parameters
        ----------
        slope : float | pint.Quantity
            the rate of frequency change in Hz/s (positive or negative) or specified unit
        """

        slope = UnitConversion.convert(slope, units.Hz/units.s, float, rounding=None)
        self.dds.set_d(SPC_DDS_CORE0_FREQ_SLOPE + self.index, float(slope))
    # aliases
    frequency_slope = freq_slope

    def get_freq_slope(self, return_unit = None) -> float:
        """
        get the frequency slope of the linearly changing frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ_SLOPE` in the manual)
        
        Parameters
        ----------
        return_unit : pint.Unit = None
            the unit of the returned frequency slope, by default None

        Returns
        -------
        float
            the rate of frequency change in Hz/s
        """

        return_value = self.dds.card.get_d(SPC_DDS_CORE0_FREQ_SLOPE + self.index)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.Hz/units.s, return_unit)
        return return_value
    # aliases
    get_frequency_slope = get_freq_slope

    def amp_slope(self, slope : float) -> None:
        """
        set the amplitude slope of the linearly changing amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP_SLOPE` in the manual)
        
        Parameters
        ----------
        slope : float | pint.Quantity
            the rate of amplitude change in 1/s (positive or negative) or specified unit
        """

        slope = UnitConversion.convert(slope, 1/units.s, float, rounding=None)
        self.dds.set_d(SPC_DDS_CORE0_AMP_SLOPE + self.index, float(slope))
    # aliases
    amplitude_slope = amp_slope

    def get_amp_slope(self, return_unit = None) -> float:
        """
        set the amplitude slope of the linearly changing amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP_SLOPE` in the manual)
        
        Parameters
        ----------
        return_unit : pint.Unit = None
            the unit of the returned amplitude slope, by default None

        Returns
        -------
        float
            the rate of amplitude change in 1/s
        """


        return_value = self.dds.card.get_d(SPC_DDS_CORE0_AMP_SLOPE + self.index)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value / units.s, return_unit)
        return return_value
    # aliases
    amplitude_slope = amp_slope


class DDS(CardFunctionality):
    """a higher-level abstraction of the SpcmCardFunctionality class to implement DDS functionality

    The DDS firmware allows the user a certain maximum number of dds cores, that 
    each on it's own generates a sine wave with the following parameters:
    * static parameters:
        + frequency
        + amplitude
        + phase
    * dynamic parameters:
        + frequency_slope
            changes the active frequency of the dds core with a linear slope
        + amplitude_slope
            changes the active amplitude of the dds core with a linear slope
    Each of these cores can either be added together and outputted, or specific groups
    of cores can be added together and outputted on a specific hardware output channel.
    Furthermore, specific dds cores can be connected to input parameters of another dds core.

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    Commands
    ---------
    The DDS functionality is controlled through commands that are listed and then written to the card.
    These written lists of commands are collected in a shadow register and are transferred to 
    the active register when a trigger is received.
    
    There are three different trigger sources, that can be set with the method 'trg_source()':
    * SPCM_DDS_TRG_SRC_NONE  = 0
        no triggers are generated and the commands are only transfered to the active register
        when a exec_now command is send
    * SPCM_DDS_TRG_SRC_TIMER = 1
        the triggers are generated on a timed grid with a period that can be set by the 
        method 'trg_timer()'
    * SPCM_DDS_TRG_SRC_CARD  = 2
        the triggers come from the card internal trigger logic (for more information, 
        see our product's user manual on how to setup the different triggers). In the DDS-mode
        multiple triggers can be processed, as with the mode SPC_STD_REP_SINGLERESTART.
    
    Note
    ----
    also the trigger source setting happens when a trigger comes. Hence a change of
    the trigger mode only happens after an 'arm()' command was send and an internal trigger was
    received.
 
    """

    cores : list[DDSCore] = []
    channels : Channels = None

    check_features : bool = False

    _current_core : int = -1
    _channel_from_core : dict[int, int] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = kwargs.get("channels", None)
        self.check_features = kwargs.get("check_features", False)
        self.cores = []
        self.load_cores()
        # Check if DDS feature is installed
        if self.check_features:
            features = self.card.ext_features()
            if not ((features & SPCM_FEAT_EXTFW_DDS20) or (features & SPCM_FEAT_EXTFW_DDS50)):
                raise SpcmException("The DDS feature is not installed on the card")
    
    def load_cores(self):
        """
        load the cores of the DDS functionality
        """

        self.cores = []
        num_cores = self.num_cores()
        
        if self.channels is not None:
            for channel in self.channels:
                cores_on_channel = self.get_cores_on_channel(channel.index)
                for core in range(num_cores):
                    if cores_on_channel & (1 << core):
                        self._channel_from_core[core] = channel
        
        for core in range(num_cores):
            if core in self._channel_from_core:
                self.cores.append(DDSCore(core, self, channel=self._channel_from_core[core]))
            else:
                self.cores.append(DDSCore(core, self))
        
    def __len__(self) -> int:
        """
        get the number of cores

        Returns
        -------
        int
            the number of cores
        """
        return len(self.cores)
    
    def __iter__(self):
        """
        make the class iterable

        Returns
        -------
        self
        """
        return self
    
    def __next__(self):
        """
        get the next core

        Returns
        -------
        DDSCore
            the next core
        """

        self._current_core += 1
        if self._current_core < len(self.cores):
            return self.cores[self._current_core]
        else:
            self._current_core = -1
            raise StopIteration
    
    def __getitem__(self, index : int) -> DDSCore:
        """
        get a specific core

        Parameters
        ----------
        index : int
            the index of the core

        Returns
        -------
        DDSCore
            the specific core
        """

        return self.cores[index]

    def set_i(self, reg : int, value : int) -> None:
        """
        set an integer value to a register

        Parameters
        ----------
        reg : int
            the register to be changed
        value : int
            the value to be set
        
        Raises
        ------
        SpcmException
            if the command list is full
        """

        self.card.set_i(reg, value)
    
    def set_d(self, reg : int, value : float) -> None:
        """
        set a double value to a register

        Parameters
        ----------
        reg : int
            the register to be changed
        value : float
            the value to be set
        """
    
        self.card.set_d(reg, value)

    def reset(self) -> None:
        """
        Resets the DDS specific part of the firmware (see register `SPC_DDS_CMD` in the manual)
        """

        self.cmd(SPCM_DDS_CMD_RESET)
    
    # DDS information
    def num_cores(self) -> int:
        """
        get the total num of available cores on the card. (see register `SPC_DDS_NUM_CORES` in the manual)

        Returns
        -------
        int
            the available number of dds cores
        """
        return self.card.get_i(SPC_DDS_NUM_CORES)
    
    def queue_cmd_max(self):
        """
        get the total number of commands that can be hold by the queue. (see register `SPC_DDS_QUEUE_CMD_MAX` in the manual)

        Returns
        -------
        int
            the total number of commands
        """
        return self.card.get_i(SPC_DDS_QUEUE_CMD_MAX)
    
    def queue_cmd_count(self):
        """
        get the current number of commands that are in the queue. (see register `SPC_DDS_QUEUE_CMD_COUNT` in the manual)

        Returns
        -------
        int
            the current number of commands
        """
        return self.card.get_i(SPC_DDS_QUEUE_CMD_COUNT)
    
    def status(self):
        return self.card.get_i(SPC_DDS_STATUS)

    # DDS setup settings
    def data_transfer_mode(self, mode : int) -> None:
        """
        set the data transfer mode for the DDS functionality (see register `SPC_DDS_DATA_TRANSFER_MODE` in the manual)

        Parameters
        ----------
        mode : int
            the data transfer mode:
            * SPCM_DDS_DTM_SINGLE = 0
                the data is transferred using single commands (with lower latency)
            * SPCM_DDS_DTM_DMA = 1
                the data is transferred using DMA (with higher bandwidth)
        """

        self._dtm = mode
        self.set_i(SPC_DDS_DATA_TRANSFER_MODE, mode)
    
    def get_data_transfer_mode(self) -> int:
        """
        get the data transfer mode for the DDS functionality (see register `SPC_DDS_DATA_TRANSFER_MODE` in the manual)

        Returns
        -------
        int
            the data transfer mode:
            * SPCM_DDS_DTM_SINGLE = 0
                the data is transferred using single commands (with lower latency)
            * SPCM_DDS_DTM_DMA = 1
                the data is transferred using DMA (with higher bandwidth)
        """

        self._dtm = self.card.get_i(SPC_DDS_DATA_TRANSFER_MODE)
        return self._dtm
    
    def phase_behaviour(self, behaviour : int) -> None:
        """
        set the phase behaviour of the DDS cores (see register `SPC_DDS_PHASE_BEHAVIOUR` in the manual)

        Parameters
        ----------
        behaviour : int
            the phase behaviour
        """

        self.set_i(SPC_DDS_PHASE_BEHAVIOUR, behaviour)
    
    def get_phase_behaviour(self) -> int:
        """
        get the phase behaviour of the DDS cores (see register `SPC_DDS_PHASE_BEHAVIOUR` in the manual)

        Returns
        -------
        int
            the phase behaviour
        """

        return self.card.get_i(SPC_DDS_PHASE_BEHAVIOUR)

    def cores_on_channel(self, channel : int, *args) -> None:
        """
        setup the cores that are connected to a specific channel (see register `SPC_DDS_CORES_ON_CH0` in the manual)

        Parameters
        ----------
        channel : int
            the channel number
        *args : int
            the cores that are connected to the channel
        
        TODO: change the channel associated with each core
        """

        mask = 0
        for core in args:
            mask |= core
        self.set_i(SPC_DDS_CORES_ON_CH0 + channel, mask)
    
    def get_cores_on_channel(self, channel : int) -> int:
        """
        get the cores that are connected to a specific channel (see register `SPC_DDS_CORES_ON_CH0` in the manual)

        Parameters
        ----------
        channel : int
            the channel number

        Returns
        -------
        int
            the cores that are connected to the channel
        """

        return self.card.get_i(SPC_DDS_CORES_ON_CH0 + channel)

    def trg_src(self, src : int) -> None:
        """
        setup the source of where the trigger is coming from (see register `SPC_DDS_TRG_SRC` in the manual)

        NOTE
        ---
        the trigger source is also set using the shadow register, hence only after an exec_at_trig or exec_now --
        
        Parameters
        ----------
        src : int
            set the trigger source:
            * SPCM_DDS_TRG_SRC_NONE  = 0
                no trigger source set, only exec_now changes what is output by the cores
            * SPCM_DDS_TRG_SRC_TIMER = 1
                an internal timer sends out triggers with a period defined by `trg_timer(period)`
            * SPCM_DDS_TRG_SRC_CARD  = 2
                use the trigger engine of the card (see the user manual for more information about setting up the trigger engine)
        """

        self.set_i(SPC_DDS_TRG_SRC, src)

    def get_trg_src(self) -> int:
        """
        get the source of where the trigger is coming from (see register `SPC_DDS_TRG_SRC` in the manual)

        NOTE
        ----
        the trigger source is also set using the shadow register, hence only after an exec_at_trig or exec_now --
        
        Returns
        ----------
        int
            get one of the trigger source:
            * SPCM_DDS_TRG_SRC_NONE  = 0
                no trigger source set, only exec_now changes what is output by the cores
            * SPCM_DDS_TRG_SRC_TIMER = 1
                an internal timer sends out triggers with a period defined by `trg_timer(period)`
            * SPCM_DDS_TRG_SRC_CARD  = 2
                use the trigger engine of the card (see the user manual for more information about setting up the trigger engine)
        """

        return self.card.get_i(SPC_DDS_TRG_SRC)
    
    def trg_timer(self, period : float) -> None:
        """
        set the period at which the timer should raise DDS trigger events. (see register `SPC_DDS_TRG_TIMER` in the manual)

        NOTE
        ----
        only used in conjecture with the trigger source set to SPCM_DDS_TRG_SRC_TIMER ---
        
        Parameters
        ----------
        period : float | pint.Quantity
            the time between DDS trigger events in seconds
        """

        period = UnitConversion.convert(period, units.s, float, rounding=None)
        self.set_d(SPC_DDS_TRG_TIMER, float(period))
    
    def get_trg_timer(self, return_unit = None) -> float:
        """
        get the period at which the timer should raise DDS trigger events. (see register `SPC_DDS_TRG_TIMER` in the manual)

        NOTE
        ----
        only used in conjecture with the trigger source set to SPCM_DDS_TRG_SRC_TIMER ---

        Parameters
        ----------
        return_unit : pint.Unit = None
            the unit of the returned time between DDS trigger events, by default None
        
        Returns
        ----------
        float
            the time between DDS trigger events in seconds
        """

        return_value = self.card.get_d(SPC_DDS_TRG_TIMER)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.s, return_unit)
        return return_value

    def x_mode(self, xio : int, mode : int) -> None:
        """
        setup the kind of output that the XIO outputs will give (see register `SPC_DDS_X0_MODE` in the manual)

        Parameters
        ----------
        xio : int
            the XIO channel number
        mode : int
            the mode that the channel needs to run in
        """

        self.set_i(SPC_DDS_X0_MODE + xio, mode)

    def get_x_mode(self, xio : int) -> int:
        """
        get the kind of output that the XIO outputs will give (see register `SPC_DDS_X0_MODE` in the manual)

        Parameters
        ----------
        xio : int
            the XIO channel number

        Returns
        -------
        int
            the mode that the channel needs to run in
            SPC_DDS_XIO_SEQUENCE = 0
                turn on and off the XIO channels using commands in the DDS cmd queue
            SPC_DDS_XIO_ARM = 1
                when the DDS firmware is waiting for a trigger to come this signal is high
            SPC_DDS_XIO_LATCH = 2
                when the DDS firmware starts executing a change this signal is high
        """

        return self.card.get_i(SPC_DDS_X0_MODE + xio)

    def freq_ramp_stepsize(self, divider : int) -> None:
        """
        number of timesteps before the frequency is changed during a frequency ramp. (see register `SPC_DDS_FREQ_RAMP_STEPSIZE` in the manual)

        NOTES
        -----
        - this is a global setting for all cores
        - internally the time divider is used to calculate the amount of change per event using a given frequency slope, please set the time divider before setting the frequency slope
        
        Parameters
        ----------
        divider : int
            the number of DDS timesteps that a value is kept constant during a frequency ramp
        """

        self.set_i(SPC_DDS_FREQ_RAMP_STEPSIZE, int(divider))

    def get_freq_ramp_stepsize(self) -> int:
        """
        get the number of timesteps before the frequency is changed during a frequency ramp. (see register `SPC_DDS_FREQ_RAMP_STEPSIZE` in the manual)
        
        NOTES
        -----
        - this is a global setting for all cores
        - internally the time divider is used to calculate the amount of change per event using a given frequency slope, please set the time divider before setting the frequency slope
        
        Returns
        ----------
        divider : int
            the number of DDS timesteps that a value is kept constant during a frequency ramp
        """

        return self.card.get_i(SPC_DDS_FREQ_RAMP_STEPSIZE)

    def amp_ramp_stepsize(self, divider : int) -> None:
        """
        number of timesteps before the amplitude is changed during a frequency ramp. (see register `SPC_DDS_AMP_RAMP_STEPSIZE` in the manual)

        NOTES
        -----
        - this is a global setting for all cores
        - internally the time divider is used to calculate the amount of change per event using a given amplitude slope, 
            please set the time divider before setting the amplitude slope
        
        Parameters
        ----------
        divider : int
            the number of DDS timesteps that a value is kept constant during an amplitude ramp
        """

        self.set_i(SPC_DDS_AMP_RAMP_STEPSIZE, int(divider))

    def get_amp_ramp_stepsize(self) -> int:
        """
        get the number of timesteps before the amplitude is changed during a frequency ramp. (see register `SPC_DDS_AMP_RAMP_STEPSIZE` in the manual)

        NOTES
        -----
        - this is a global setting for all cores
        - internally the time divider is used to calculate the amount of change per event using a given amplitude slope, 
            please set the time divider before setting the amplitude slope
        
        Returns
        ----------
        divider : int
            the number of DDS timesteps that a value is kept constant during an amplitude ramp
        """

        return self.card.get_i(SPC_DDS_AMP_RAMP_STEPSIZE)

    # DDS "static" parameters
    # def amp(self, core_index : int, amplitude : float) -> None:
    def amp(self, *args) -> None:
        """
        set the amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP` in the manual)
        
        Parameters
        ----------
        core_index : int (optional)
            the index of the core to be changed
        amplitude : float
            the value between 0 and 1 corresponding to the amplitude
        """

        if len(args) == 1:
            amplitude = args[0]
            for core in self.cores:
                core.amp(amplitude)
        elif len(args) == 2:
            core_index, amplitude = args
            self.cores[core_index].amp(amplitude)
        else:
            raise TypeError("amp() takes 1 or 2 positional arguments ({} given)".format(len(args) + 1))
        # self.set_d(SPC_DDS_CORE0_AMP + core_index, float(amplitude))
    # aliases
    amplitude = amp

    def get_amp(self, core_index : int, return_unit = None) -> float:
        """
        gets the amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        return_unit : pint.Unit = None
            the unit of the returned amplitude, by default None

        Returns
        -------
        float | pint.Quantity
            the value between 0 and 1 corresponding to the amplitude of the specific core or in the specified unit
        """

        return self.cores[core_index].get_amp(return_unit)
        # return self.card.get_d(SPC_DDS_CORE0_AMP + core_index)
    # aliases
    get_amplitude = get_amp

    def avail_amp_min(self) -> float:
        """
        get the minimum available amplitude (see register `SPC_DDS_AVAIL_AMP_MIN` in the manual)

        Returns
        -------
        float
            the minimum available amplitude
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_MIN)
    
    def avail_amp_max(self) -> float:
        """
        get the maximum available amplitude (see register `SPC_DDS_AVAIL_AMP_MAX` in the manual)

        Returns
        -------
        float
            the maximum available amplitude
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_MAX)
    
    def avail_amp_step(self) -> float:
        """
        get the step size of the available amplitudes (see register `SPC_DDS_AVAIL_AMP_STEP` in the manual)

        Returns
        -------
        float
            the step size of the available amplitudes
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_STEP)

    # def freq(self, core_index : int, frequency : float) -> None:
    def freq(self, *args) -> None:
        """
        set the frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ` in the manual)
        
        Parameters
        ----------
        core_index : int (optional)
            the index of the core to be changed
        frequency : float
            the value of the frequency in Hz
        """

        if len(args) == 1:
            frequency = args[0]
            for core in self.cores:
                core.freq(frequency)
        elif len(args) == 2:
            core_index, frequency = args
            self.cores[core_index].freq(frequency)
        else:
            raise TypeError("freq() takes 1 or 2 positional arguments ({} given)".format(len(args) + 1))
        # self.set_d(SPC_DDS_CORE0_FREQ + core_index, float(frequency))
    # aliases
    frequency = freq

    def get_freq(self, core_index : int, return_unit = None) -> float:
        """
        gets the frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        return_unit : pint.Unit = None
            the unit of the returned frequency, by default None
        
        Returns
        -------
        float | pint.Quantity
            the value of the frequency in Hz the specific core or in the specified unit
        """

        return self.cores[core_index].get_freq(return_unit)
    # aliases
    get_frequency = get_freq

    def avail_freq_min(self) -> float:
        """
        get the minimum available frequency (see register `SPC_DDS_AVAIL_FREQ_MIN` in the manual)

        Returns
        -------
        float
            the minimum available frequency
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_MIN)
    
    def avail_freq_max(self) -> float:
        """
        get the maximum available frequency (see register `SPC_DDS_AVAIL_FREQ_MAX` in the manual)

        Returns
        -------
        float
            the maximum available frequency
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_MAX)
    
    def avail_freq_step(self) -> float:
        """
        get the step size of the available frequencies (see register `SPC_DDS_AVAIL_FREQ_STEP` in the manual)

        Returns
        -------
        float
            the step size of the available frequencies
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_STEP)

    # def phase(self, core_index : int, phase : float) -> None:
    def phase(self, *args) -> None:
        """
        set the phase of the sine wave of a specific core (see register `SPC_DDS_CORE0_PHASE` in the manual)
        
        Parameters
        ----------
        core_index : int (optional)
            the index of the core to be changed
        phase : float
            the value between 0 and 360 degrees of the phase
        """

        if len(args) == 1:
            phase = args[0]
            for core in self.cores:
                core.phase(phase)
        elif len(args) == 2:
            core_index, phase = args
            self.cores[core_index].phase(phase)
        else:
            raise TypeError("phase() takes 1 or 2 positional arguments ({} given)".format(len(args) + 1))
        # self.set_d(SPC_DDS_CORE0_PHASE + core_index, float(phase))

    def get_phase(self, core_index : int, return_unit = None) -> float:
        """
        gets the phase of the sine wave of a specific core (see register `SPC_DDS_CORE0_PHASE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        return_unit : pint.Unit = None
            the unit of the returned phase, by default None
        
        Returns
        -------
        float
            the value between 0 and 360 degrees of the phase
        """

        return self.cores[core_index].get_phase(return_unit)
      
    def avail_phase_min(self) -> float:
        """
        get the minimum available phase (see register `SPC_DDS_AVAIL_PHASE_MIN` in the manual)

        Returns
        -------
        float
            the minimum available phase
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_PHASE_MIN)
    
    def avail_phase_max(self) -> float:
        """
        get the maximum available phase (see register `SPC_DDS_AVAIL_PHASE_MAX` in the manual)

        Returns
        -------
        float
            the maximum available phase
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_PHASE_MAX)
    
    def avail_phase_step(self) -> float:
        """
        get the step size of the available phases (see register `SPC_DDS_AVAIL_PHASE_STEP` in the manual)

        Returns
        -------
        float
            the step size of the available phases
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_PHASE_STEP)

    def x_manual_output(self, state_mask : int) -> None:
        """
        set the output of the xio channels using a bit mask (see register `SPC_DDS_X_MANUAL_OUTPUT` in the manual)
        
        Parameters
        ----------
        state_mask : int
            bit mask where the bits correspond to specific channels and 1 to on and 0 to off.
        """

        self.set_i(SPC_DDS_X_MANUAL_OUTPUT, state_mask)

    def get_x_manual_output(self) -> int:
        """
        get the output of the xio channels using a bit mask (see register `SPC_DDS_X_MANUAL_OUTPUT` in the manual)
        
        Returns
        ----------
        int
            bit mask where the bits correspond to specific channels and 1 to on and 0 to off.
        """

        return self.card.get_i(SPC_DDS_X_MANUAL_OUTPUT)

    # DDS dynamic parameters
    # def freq_slope(self, core_index : int, slope : float) -> None:
    def freq_slope(self, *args) -> None:
        """
        set the frequency slope of the linearly changing frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int (optional)
            the index of the core to be changed
        slope : float
            the rate of frequency change in Hz/s
        """

        if len(args) == 1:
            slope = args[0]
            for core in self.cores:
                core.freq_slope(slope)
        elif len(args) == 2:
            core_index, slope = args
            self.cores[core_index].freq_slope(slope)
        else:
            raise TypeError("freq_slope() takes 1 or 2 positional arguments ({} given)".format(len(args) + 1))
        # self.set_d(SPC_DDS_CORE0_FREQ_SLOPE + core_index, float(slope))
    # aliases
    frequency_slope = freq_slope

    def get_freq_slope(self, core_index : int, return_unit=None) -> float:
        """
        get the frequency slope of the linearly changing frequency of the sine wave of a specific core (see register `SPC_DDS_CORE0_FREQ_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        return_unit : pint.Unit = None
            the unit of the returned frequency slope, by default None
        
        Returns
        -------
        float
            the rate of frequency change in Hz/s
        """

        return self.cores[core_index].get_freq_slope(return_unit)
    # aliases
    get_frequency_slope = get_freq_slope

    def avail_freq_slope_min(self) -> float:
        """
        get the minimum available frequency slope (see register `SPC_DDS_AVAIL_FREQ_SLOPE_MIN` in the manual)

        Returns
        -------
        float
            the minimum available frequency slope
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_SLOPE_MIN)
    
    def avail_freq_slope_max(self) -> float:
        """
        get the maximum available frequency slope (see register `SPC_DDS_AVAIL_FREQ_SLOPE_MAX` in the manual)

        Returns
        -------
        float
            the maximum available frequency slope
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_SLOPE_MAX)
    
    def avail_freq_slope_step(self) -> float:
        """
        get the step size of the available frequency slopes (see register `SPC_DDS_AVAIL_FREQ_SLOPE_STEP` in the manual)

        Returns
        -------
        float
            the step size of the available frequency slopes
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_FREQ_SLOPE_STEP)

    # def amp_slope(self, core_index : int, slope : float) -> None:
    def amp_slope(self, *args) -> None:
        """
        set the amplitude slope of the linearly changing amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int (optional)
            the index of the core to be changed
        slope : float
            the rate of amplitude change in 1/s
        """

        if len(args) == 1:
            slope = args[0]
            for core in self.cores:
                core.amp_slope(slope)
        elif len(args) == 2:
            core_index, slope = args
            self.cores[core_index].amp_slope(slope)
        else:
            raise TypeError("amp_slope() takes 1 or 2 positional arguments ({} given)".format(len(args) + 1))
        # self.set_d(SPC_DDS_CORE0_AMP_SLOPE + core_index, float(slope))
    # aliases
    amplitude_slope = amp_slope

    def get_amp_slope(self, core_index : int, return_unit = None) -> float:
        """
        set the amplitude slope of the linearly changing amplitude of the sine wave of a specific core (see register `SPC_DDS_CORE0_AMP_SLOPE` in the manual)
        
        Parameters
        ----------
        core_index : int
            the index of the core to be changed
        return_unit : pint.Unit = None
            the unit of the returned amplitude slope, by default None
        
        Returns
        -------
        float
            the rate of amplitude change in 1/s
        """

        return self.cores[core_index].get_amp_slope(return_unit)
    # aliases
    amplitude_slope = amp_slope

    def avail_amp_slope_min(self) -> float:
        """
        get the minimum available amplitude slope (see register `SPC_DDS_AVAIL_AMP_SLOPE_MIN` in the manual)

        Returns
        -------
        float
            the minimum available amplitude slope
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_SLOPE_MIN)
    
    def avail_amp_slope_max(self) -> float:
        """
        get the maximum available amplitude slope (see register `SPC_DDS_AVAIL_AMP_SLOPE_MAX` in the manual)

        Returns
        -------
        float
            the maximum available amplitude slope
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_SLOPE_MAX)
    
    def avail_amp_slope_step(self) -> float:
        """
        get the step size of the available amplitude slopes (see register `SPC_DDS_AVAIL_AMP_SLOPE_STEP` in the manual)

        Returns
        -------
        float
            the step size of the available amplitude slopes
        
        TODO: unitize!
        """

        return self.card.get_d(SPC_DDS_AVAIL_AMP_SLOPE_STEP)

    # DDS control
    def cmd(self, command : int) -> None:
        """
        execute a DDS specific control flow command (see register `SPC_DDS_CMD` in the manual)
        
        Parameters
        ----------
        command : int
            DDS specific command
        """

        self.set_i(SPC_DDS_CMD, command)

    def exec_at_trg(self) -> None:
        """
        execute the commands in the shadow register at the next trigger event (see register `SPC_DDS_CMD` in the manual)
        """
        self.cmd(SPCM_DDS_CMD_EXEC_AT_TRG)
    # aliases
    arm = exec_at_trg
    wait_for_trg = exec_at_trg
    
    def exec_now(self) -> None:
        """
        execute the commands in the shadow register as soon as possible (see register `SPC_DDS_CMD` in the manual)
        """

        self.cmd(SPCM_DDS_CMD_EXEC_NOW)
    # aliases
    direct_latch = exec_now

    def trg_count(self) -> int:
        """
        get the number of trigger exec_at_trg and exec_now command that have been executed (see register `SPC_DDS_TRG_COUNT` in the manual)

        Returns
        -------
        int
            the number of trigger exec_at_trg and exec_now command that have been executed
        """

        return self.card.get_i(SPC_DDS_TRG_COUNT)
    
    def write_to_card(self, flags=0) -> None:
        """
        send a list of all the commands that came after the last write_list and send them to the card (see register `SPC_DDS_CMD` in the manual)

        Parameters
        ----------
        flags : int = 0
            the flags that can be set with the write_to_card command
        """
        
        self.cmd(SPCM_DDS_CMD_WRITE_TO_CARD | flags)
    
    # DDS helper functions
    def kwargs2mask(self, kwargs : dict[str, bool], prefix : str = "") -> int:
        """
        DDS helper: transform a dictionary with keys with a specific prefix to a bitmask

        Parameters
        ----------
        kwargs : dict
            dictonary with keys with a specific prefix and values given by bools
        prefix : str
            a prefix for the key names
        
        Returns
        -------
        int
            bit mask
        
        Example
        -------
        ['core_0' = True, 'core_2' = False, 'core_3' = True] => 0b1001 = 9
        """
        
        mask = 0
        for keyword, value in kwargs.items():
            bit = int(keyword[len(prefix):])
            if value:
                mask |= 1 << bit
            else:
                mask &= ~(1 << bit)
        return mask
    # aliases
    k2m = kwargs2mask