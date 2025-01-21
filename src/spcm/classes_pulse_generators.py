# -*- coding: utf-8 -*-

from spcm_core.constants import *

import numpy as np

from .classes_error_exception import SpcmException
from .classes_card import Card
from .classes_functionality import CardFunctionality

import pint
from .classes_unit_conversion import UnitConversion
from . import units

class PulseGenerator:
    """
    a class to implement a single pulse generator
    
    Parameters
    ----------
    card : Card
        the card object that is used by the functionality
    pg_index : int
        the index of the used pulse generator
    """

    card : Card
    pg_index : int
    """The index of the pulse generator"""

    _reg_distance : int = 100

    def __init__(self, card : Card, pg_index : int, *args, **kwargs) -> None:
        """
        The constructor of the PulseGenerator class

        Parameters
        ----------
        card : Card
            the card object that is used by the functionality
        pg_index : int
            the index of the used pulse generator
        """
        
        self.card = card
        self.pg_index = pg_index

    def __str__(self) -> str:
        """
        String representation of the PulseGenerator class
    
        Returns
        -------
        str
            String representation of the PulseGenerator class
        """
        
        return f"PulseGenerator(card={self.card}, pg_index={self.pg_index})"
    
    __repr__ = __str__
    
    # The trigger behavior of the pulse generator
    def mode(self, mode : int = None) -> int:
        """
        Set the trigger mode of the pulse generator (see register 'SPC_XIO_PULSEGEN0_MODE' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        mode : int
            The trigger mode
        
        Returns
        -------
        int
            The trigger mode
        """

        if mode is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_MODE + self._reg_distance*self.pg_index, mode)
        return self.card.get_i(SPC_XIO_PULSEGEN0_MODE + self._reg_distance*self.pg_index)
    
    # The duration of a single period
    def period_length(self, length : int = None) -> int:
        """
        Set the period length of the pulse generator (see register 'SPC_XIO_PULSEGEN0_LEN' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        length : int
            The period length in clock cycles
        
        Returns
        -------
        int
            The period length in clock cycles
        """

        if length is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_LEN + self._reg_distance*self.pg_index, length)
        return self.card.get_i(SPC_XIO_PULSEGEN0_LEN + self._reg_distance*self.pg_index)

    def avail_length_min(self) -> int:
        """
        Returns the minimum length (period) of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILLEN_MIN' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available minimal length in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLEN_MIN)
    
    def avail_length_max(self) -> int:
        """
        Returns the maximum length (period) of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILLEN_MAX' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available maximal length in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLEN_MAX)
    
    def avail_length_step(self) -> int:
        """
        Returns the step size of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILLEN_STEP' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available step size in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLEN_STEP)
    
    # The time that the signal is high during one period
    def high_length(self, length : int = None) -> int:
        """
        Set the high length of the pulse generator (see register 'SPC_XIO_PULSEGEN0_HIGH' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        pg_index : int
            The index of the pulse generator
        length : int
            The high length in clock cycles
        
        Returns
        -------
        int
            The high length in clock cycles
        """

        if length is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_HIGH + self._reg_distance*self.pg_index, length)
        return self.card.get_i(SPC_XIO_PULSEGEN0_HIGH + self._reg_distance*self.pg_index)
    
    def avail_high_min(self) -> int:
        """
        Returns the minimum high length of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILHIGH_MIN' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available minimal high length in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILHIGH_MIN)
    
    def avail_high_max(self) -> int:
        """
        Returns the maximum high length of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILHIGH_MAX' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available maximal high length in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILHIGH_MAX)
    
    def avail_high_step(self) -> int:
        """
        Returns the step size of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILHIGH_STEP' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available step size in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILHIGH_STEP)
    
    # The number of times that a single period is repeated
    def num_loops(self, loops : int = None) -> int:
        """
        Set the number of loops of a single period on the pulse generator (see register 'SPC_XIO_PULSEGEN0_LOOPS' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        loops : int
            The number of loops
        
        Returns
        -------
        int
            The number of loops
        """

        if loops is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_LOOPS + self._reg_distance*self.pg_index, loops)
        return self.card.get_i(SPC_XIO_PULSEGEN0_LOOPS + self._reg_distance*self.pg_index)
    
    def avail_loops_min(self) -> int:
        """
        Returns the minimum number of loops of the pulse generator’s output pulses. (see register 'SPC_XIO_PULSEGEN_AVAILLOOPS_MIN' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available minimal number of loops
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLOOPS_MIN)
    
    def avail_loops_max(self) -> int:
        """
        Returns the maximum number of loops of the pulse generator’s output pulses. (see register 'SPC_XIO_PULSEGEN_AVAILLOOPS_MAX' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available maximal number of loops
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLOOPS_MAX)
    
    def avail_loops_step(self) -> int:
        """
        Returns the step size of the pulse generator’s output pulses. (see register 'SPC_XIO_PULSEGEN_AVAILLOOPS_STEP' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            Returns the step size when defining the repetition of pulse generator’s output.
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILLOOPS_STEP)
    
    # The delay between the start of the pulse generator and the first pulse
    def delay(self, delay : int = None) -> int:
        """
        Set the delay of the pulse generator (see register 'SPC_XIO_PULSEGEN0_DELAY' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        delay : int
            The delay in clock cycles
        
        Returns
        -------
        int
            The delay in clock cycles
        """

        if delay is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_DELAY + self._reg_distance*self.pg_index, delay)
        return self.card.get_i(SPC_XIO_PULSEGEN0_DELAY + self._reg_distance*self.pg_index)
    
    def avail_delay_min(self) -> int:
        """
        Returns the minimum delay of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILDELAY_MIN' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available minimal delay in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILDELAY_MIN)
    
    def avail_delay_max(self) -> int:
        """
        Returns the maximum delay of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILDELAY_MAX' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available maximal delay in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILDELAY_MAX)
    
    def avail_delay_step(self) -> int:
        """
        Returns the step size of the delay of the pulse generator’s output pulses in clock cycles. (see register 'SPC_XIO_PULSEGEN_AVAILDELAY_STEP' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The available step size of the delay in clock cycles
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_AVAILDELAY_STEP)

    # Trigger muxes
    def mux1(self, mux : int = None) -> int:
        """
        Set the trigger mux 1 of the pulse generator (see register 'SPC_XIO_PULSEGEN0_MUX1_SRC' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        mux : int
            The trigger mux 1
        
        Returns
        -------
        int
            The trigger mux 1
        """

        if mux is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_MUX1_SRC + self._reg_distance*self.pg_index, mux)
        return self.card.get_i(SPC_XIO_PULSEGEN0_MUX1_SRC + self._reg_distance*self.pg_index)
    
    def mux2(self, mux : int = None) -> int:
        """
        Set the trigger mux 2 of the pulse generator (see register 'SPC_XIO_PULSEGEN0_MUX2_SRC' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        mux : int
            The trigger mux 2

        Returns
        -------
        int
            The trigger mux 2
        """

        if mux is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_MUX2_SRC + self._reg_distance*self.pg_index, mux)
        return self.card.get_i(SPC_XIO_PULSEGEN0_MUX2_SRC + self._reg_distance*self.pg_index)
    
    def config(self, config : int = None) -> int:
        """
        Set the configuration of the pulse generator (see register 'SPC_XIO_PULSEGEN0_CONFIG' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        config : int
            The configuration of the pulse generator

        Returns
        -------
        int
            The configuration of the pulse generator
        """

        if config is not None:
            self.card.set_i(SPC_XIO_PULSEGEN0_CONFIG + self._reg_distance*self.pg_index, config)
        return self.card.get_i(SPC_XIO_PULSEGEN0_CONFIG + self._reg_distance*self.pg_index)
    
    def _get_clock(self, return_unit : pint.Unit = None) -> int:
        """
        Get the clock rate of the pulse generator (see register 'SPC_XIO_PULSEGEN_CLOCK' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The clock rate in Hz
        """

        return_value = self.card.get_i(SPC_XIO_PULSEGEN_CLOCK)
        return_value = UnitConversion.to_unit(return_value * units.Hz, return_unit)
        return return_value
    
    # Higher abtraction functions

    def pulse_period(self, period : pint.Quantity = None, return_unit : pint.Unit = units.s) -> pint.Quantity:
        """
        Set the period length of the pulse generator signal in a time unit

        Parameters
        ----------
        period : pint.Quantity
            The period length in seconds
        
        Returns
        -------
        pint.Quantity
            The period length in seconds
        """

        if period is not None:
            if isinstance(period, pint.Quantity):
                period = int((period * self._get_clock(units.Hz)).to_base_units().magnitude)
            else:
                raise ValueError("The period must be a pint.Quantity")
            self.period_length(period)
        return_value = self.period_length()
        return_value = UnitConversion.to_unit((return_value / self._get_clock(units.Hz)), return_unit)
        return return_value
    
    def repetition_rate(self, rate : pint.Quantity = None, return_unit : pint.Unit = units.Hz) -> pint.Quantity:
        """
        Set the repetition rate of the pulse generator signal in a frequency unit

        Parameters
        ----------
        rate : pint.Quantity
            The repetition rate in Hz
        
        Returns
        -------
        pint.Quantity
            The repetition rate in Hz
        """

        if rate is not None:
            if isinstance(rate, pint.Quantity):
                period = int(np.rint((self._get_clock(units.Hz) / rate).to_base_units().magnitude))
            else:
                raise ValueError("The rate must be a pint.Quantity")
            self.period_length(period)
        return_value = self.period_length()
        return_value = UnitConversion.to_unit((self._get_clock(units.Hz) / return_value), return_unit)
        return return_value
    
    def pulse_length(self, length : pint.Quantity, return_unit : pint.Unit = units.s) -> pint.Quantity:
        """
        Set the pulse length of the pulse generator signal in a time unit

        Parameters
        ----------
        length : pint.Quantity
            The pulse length in seconds
        
        Returns
        -------
        pint.Quantity
            The pulse length in seconds
        """

        if length is not None:
            if isinstance(length, pint.Quantity):
                length = int((length * self._get_clock(units.Hz)).to_base_units().magnitude)
            else:
                raise ValueError("The length must be a pint.Quantity")
            self.high_length(length)
        return_value = self.high_length()
        return_value = UnitConversion.to_unit((return_value / self._get_clock(units.Hz)), return_unit)
        return return_value
    
    def duty_cycle(self, duty_cycle : pint.Quantity = None, return_unit : pint.Unit = units.percent) -> pint.Quantity:
        """
        Set the duty cycle of the pulse generator signal in a percentage unit

        Parameters
        ----------
        duty_cycle : pint.Quantity
            The duty cycle in percentage
        
        Returns
        -------
        pint.Quantity
            The duty cycle in percentage
        """

        period_length = self.period_length()
        if duty_cycle is not None:
            if isinstance(duty_cycle, pint.Quantity):
                high_length = int(np.rint(period_length * duty_cycle))
            else:
                raise ValueError("The cycle must be a pint.Quantity")
            self.high_length(high_length)
        return_value = self.high_length()
        return_value = UnitConversion.to_unit((return_value / period_length) * 100 * units.percent, return_unit)
        return return_value
    
    def start_delay(self, delay : pint.Unit = None, return_unit : pint.Unit = units.s) -> pint.Unit:
        """
        Set the start delay of the pulse generator signal in a time unit

        Parameters
        ----------
        delay : pint.Unit
            The start delay in a pint quantity with time unit
        
        Returns
        -------
        pint.Unit
            The start delay in a pint quantity with time unit
        """

        if delay is not None:
            if isinstance(delay, pint.Quantity):
                delay = int((delay * self._get_clock(units.Hz)).to_base_units().magnitude)
            else:
                raise ValueError("The delay must be a pint.Quantity")
        return_value = self.delay(delay)
        return_value = UnitConversion.to_unit((return_value / self._get_clock(units.Hz)), return_unit)
        return return_value
    
    repetitions = num_loops

    def start_condition_state_signal(self, signal : int = 0, invert : bool = False) -> int:
        """
        Set the start condition state signal of the pulse generator (see register 'SPC_XIO_PULSEGEN0_MUX1' in chapter `Pulse Generator` in the manual)

        NOTE
        ----
        The Pulse Generator is started when the combined signal of both start condition signals are true and a rising edge
        is detected. The invert parameter inverts the start condition state signal.
        
        Parameters
        ----------
        signal : int
            The start condition state signal
        invert : bool
            Invert the start condition state signal
        
        Returns
        -------
        int
            The start condition state signal
        """

        return_signal = self.mux1(signal)
        return_invert = self.config()
        if invert:
            return_invert |= SPCM_PULSEGEN_CONFIG_MUX1_INVERT
        else:
            return_invert &= ~SPCM_PULSEGEN_CONFIG_MUX1_INVERT
        return_invert = self.config(return_invert)
        return return_signal, ((return_invert & SPCM_PULSEGEN_CONFIG_MUX1_INVERT) != 0)

    def start_condition_trigger_signal(self, signal : int = 0, invert : bool = False) -> int:
        """
        Set the start condition trigger signal of the pulse generator (see register 'SPC_XIO_PULSEGEN0_MUX2' in chapter `Pulse Generator` in the manual)

        NOTE
        ----
        The Pulse Generator is started when the combined signal of both start condition signals are true and a rising edge
        is detected. The invert parameter inverts the start condition state signal.
        
        Parameters
        ----------
        signal : int
            The start condition trigger signal
        invert : bool
            Invert the start condition trigger signal
        
        Returns
        -------
        int
            The start condition trigger signal
        """

        return_signal = self.mux2(signal)
        return_invert = self.config()
        if invert:
            return_invert |= SPCM_PULSEGEN_CONFIG_MUX2_INVERT
        else:
            return_invert &= ~SPCM_PULSEGEN_CONFIG_MUX2_INVERT
        return_invert = self.config(return_invert)
        return return_signal, ((return_invert & SPCM_PULSEGEN_CONFIG_MUX2_INVERT) != 0)
    
    def invert_start_condition(self, invert : bool = None) -> bool:
        """
        Invert the start condition of the pulse generator

        Parameters
        ----------
        invert : bool
            Invert the start condition
        
        Returns
        -------
        bool
            The start condition inversion
        """

        if invert is not None:
            return_invert = self.config()
            if invert:
                return_invert |= SPCM_PULSEGEN_CONFIG_INVERT
            else:
                return_invert &= ~SPCM_PULSEGEN_CONFIG_INVERT
            self.config(return_invert)
        return ((self.config() & SPCM_PULSEGEN_CONFIG_INVERT) != 0)

class PulseGenerators(CardFunctionality):
    """
    a higher-level abstraction of the CardFunctionality class to implement Pulse generator functionality

    Parameters
    ----------
    generators : list[PulseGenerator]
        a list of pulse generators
    num_generators : int
        the number of pulse generators on the card
    """
    
    generators : list[PulseGenerator]
    num_generators = 4

    def __init__(self, card : Card, enable : int = 0, *args, **kwargs) -> None:
        """
        The constructor of the PulseGenerators class

        Parameters
        ----------
        card : Card
            the card object that is used by the functionality
        enable : int or bool
            Enable or disable (all) the different pulse generators, by default all are turned off
        
        Raises
        ------
        SpcmException
        """

        super().__init__(card, *args, **kwargs)
        # Check for the pulse generator option on the card
        features = self.card.get_i(SPC_PCIEXTFEATURES)
        if features & SPCM_FEAT_EXTFW_PULSEGEN:
            self.card._print(f"Pulse generator option available")
        else:
            raise SpcmException(text="This card doesn't have the pulse generator functionality installed. Please contact sales@spec.de to get more information about this functionality.")
        
        self.load()
        self.enable(enable)
    
    def load(self) -> None:
        """Load the pulse generators"""
        self.generators = [PulseGenerator(self.card, i) for i in range(self.num_generators)]

    # TODO in the next driver release there will be a register to get the number of pulse generators
    def get_num_generators(self) -> int:
        """
        Get the number of pulse generators on the card

        Returns
        -------
        int
            The number of pulse generators
        """

        return self.num_generators
    
    def __str__(self) -> str:
        """
        String representation of the PulseGenerators class
    
        Returns
        -------
        str
            String representation of the PulseGenerators class
        """
        
        return f"PulseGenerators(card={self.card})"
    
    __repr__ = __str__
    

    def __iter__(self) -> "PulseGenerators":
        """Define this class as an iterator"""
        return self
    
    def __getitem__(self, index : int) -> PulseGenerator:
        """
        Get the pulse generator at the given index

        Parameters
        ----------
        index : int
            The index of the pulse generator
        
        Returns
        -------
        PulseGenerator
            The pulse generator at the given index
        """

        return self.generators[index]
    
    _generator_iterator_index = -1
    def __next__(self) -> PulseGenerator:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        PulseGenerator
            the next available pulse generator
        
        Raises
        ------
        StopIteration
        """
        self._generator_iterator_index += 1
        if self._generator_iterator_index >= self.num_generators:
            self._generator_iterator_index = -1
            raise StopIteration
        return self.generators[self._generator_iterator_index]
    
    def __len__(self) -> int:
        """Returns the number of available pulse generators"""
        return len(self.generators)

    # The pulse generator can be enabled or disabled
    def enable(self, enable : int = None) -> int:
        """
        Enable or disable (all) the pulse generators (see register 'SPC_XIO_PULSEGEN_ENABLE' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        enable : int or bool
            Enable or disable (all) the different pulse generators, by default all are turned off
        
        Returns
        -------
        int
            The enable state of the pulse generators
        """

        
        if isinstance(enable, bool):
            enable = (1 << self.num_generators) - 1 if enable else 0
        if enable is not None:
            self.card.set_i(SPC_XIO_PULSEGEN_ENABLE, enable)
        return self.card.get_i(SPC_XIO_PULSEGEN_ENABLE)
    
    def cmd(self, cmd : int) -> None:
        """
        Execute a command on the pulse generator (see register 'SPC_XIO_PULSEGEN_COMMAND' in chapter `Pulse Generator` in the manual)

        Parameters
        ----------
        cmd : int
            The command to execute
        """
        self.card.set_i(SPC_XIO_PULSEGEN_COMMAND, cmd)
    
    def force(self) -> None:
        """
        Generate a single rising edge, that is common for all pulse generator engines. This allows to start/trigger the output 
        of all enabled pulse generators synchronously by issuing a software command. (see register 'SPC_XIO_PULSEGEN_COMMAND' in chapter `Pulse Generator` in the manual)
        """
        self.cmd(SPCM_PULSEGEN_CMD_FORCE)
    
    def write_setup(self) -> None:
        """Write the setup of the pulse generator to the card"""
        self.card.write_setup()
    
    # The clock rate in Hz of the pulse generator
    def get_clock(self) -> int:
        """
        Get the clock rate of the pulse generator (see register 'SPC_XIO_PULSEGEN_CLOCK' in chapter `Pulse Generator` in the manual)

        Returns
        -------
        int
            The clock rate in Hz
        """
        return self.card.get_i(SPC_XIO_PULSEGEN_CLOCK)
    

    