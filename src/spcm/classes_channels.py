# -*- coding: utf-8 -*-

from spcm_core.constants import *

import numpy as np
import numpy.typing as npt

from .classes_card import Card
from .classes_card_stack import CardStack

from .classes_error_exception import SpcmException

from .classes_unit_conversion import UnitConversion
from . import units
import pint

class Channel:
    """A class to represent a channel of a card only used inside the Channels class in the list of channels"""

    card : Card = None
    index : int = 0
    data_index : int = 0

    _conversion_amp : pint.Quantity = None
    _conversion_offset : pint.Quantity = None
    _output_load : pint.Quantity = None
    _series_impedance : pint.Quantity = None

    def __init__(self, index : int, data_index : int, card : Card) -> None:
        """
        Constructor of the Channel class
    
        Parameters
        ----------
        index : int
            The index of the channel
        card : Card
            The card of the channel
        """

        self.card = card
        self.index = index
        self.data_index = data_index
        self._conversion_amp = None
        self._conversion_offset = 0 * units.percent
        self._output_load = 50 * units.ohm
        self._series_impedance = 50 * units.ohm
    
    def __str__(self) -> str:
        """
        String representation of the Channel class
    
        Returns
        -------
        str
            String representation of the Channel class
        """
        
        return f"Channel {self.index}"
    
    __repr__ = __str__

    def __int__(self) -> int:
        """
        The Channel object acts like an int and returns the index of the channel and can also be used as the index in an array

        Returns
        -------
        int
            The index of the channel
        """
        return self.data_index
    __index__ = __int__
    
    def __add__(self, other):
        """
        The Channel object again acts like an int and returns the index of the channel plus the other value
        
        Parameters
        ----------
        other : int or float
            The value to be added to the index of the channel
        
        Returns
        -------
        int or float
            The index of the channel plus the other value
        """
        return self.index + other
    
    def enable(self, enable : bool = None) -> bool:
        """
        Enables the analog front-end of the channel of the card (see register `SPC_ENABLEOUT` in the manual)
    
        Parameters
        ----------
        enable : bool
            Turn-on (True) or off (False) the spezific channel

        Returns
        -------
        bool
            The enable state of the specific channel
        """

        if enable is not None:
            self.card.set_i(SPC_ENABLEOUT0 + (SPC_ENABLEOUT1 - SPC_ENABLEOUT0) * self.index, int(enable))
        return bool(self.card.get_i(SPC_ENABLEOUT0 + (SPC_ENABLEOUT1 - SPC_ENABLEOUT0) * self.index))
    enable_out = enable
    
    def path(self, value : int = None) -> int:
        """
        Sets the input path of the channel of the card (see register `SPC_PATH0` in the manual).
        To make the read back of the coupling offset compensation setting possible, we also set 
        the register `SPC_READAIPATH` to the same value.
    
        Parameters
        ----------
        value : int
            The input path of the specific channel
        
        Returns
        -------
        int
            The input path of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_PATH0 + (SPC_PATH1 - SPC_PATH0) * self.index, value)
            self.card.set_i(SPC_READAIPATH, value)
        return self.card.get_i(SPC_PATH0 + (SPC_PATH1 - SPC_PATH0) * self.index)
    
    def amp(self, value : int = None, return_unit = None) -> int:
        """
        Sets the output/input range (amplitude) of the analog front-end of the channel of the card in mV (see register `SPC_AMP` in the manual)
    
        Parameters
        ----------
        value : int
            The output range (amplitude) of the specific channel in millivolts
        unit : pint.Unit = None
            The unit of the return value
            
        Returns
        -------
        int | pint.Quantity
            The output range (amplitude) of the specific channel in millivolts or the unit specified
        """

        if value is not None:
            if isinstance(value, pint.Quantity):
                value = self.voltage_conversion(value)
            self._conversion_amp = UnitConversion.force_unit(value, units.mV)
            value = UnitConversion.convert(value, units.mV, int)
            self.card.set_i(SPC_AMP0 + (SPC_AMP1 - SPC_AMP0) * self.index, value)
        value = self.card.get_i(SPC_AMP0 + (SPC_AMP1 - SPC_AMP0) * self.index)
        value = UnitConversion.to_unit(value * units.mV, return_unit)
        return value

    def offset(self, value : int = None, return_unit = None) -> int:
        """
        Sets the offset of the analog front-end of the channel of the card in % of the full range o rmV (see register `SPC_OFFS0` in the manual)
        If the value is given and has a unit, then this unit is converted to the unit of the card (mV or %)
    
        Parameters
        ----------
        value : int | pint.Quantity = None
            The offset of the specific channel as integer in % or as a Quantity in % or mV
        unit : pint.Unit = None
            The unit of the return value
            
        Returns
        -------
        int | pint.Quantity
            The offset of the specific channel in % or the unit specified by return_unit
        """

        # Analog in cards are programmed in percent of the full range and analog output cards in mV (in the M2p, M4i/x and M5i families)
        card_unit = 1
        fnc_type = self.card.function_type()
        if fnc_type == SPCM_TYPE_AI:
            card_unit = units.percent
        elif fnc_type == SPCM_TYPE_AO:
            card_unit = units.mV

        if value is not None:
            # The user gives a value as a Quantity
            if isinstance(value, pint.Quantity):
                if fnc_type == SPCM_TYPE_AO:
                    # The card expects a value in mV
                    if value.check('[]'):
                        # Convert from percent to mV
                        value = (value * self._conversion_amp).to(card_unit)
                    else:
                        value = value.to(card_unit)
                elif fnc_type == SPCM_TYPE_AI:
                    # The card expects a value in percent
                    if value.check('[electric_potential]'):
                        # Convert from mV to percent
                        value = (value / self._conversion_amp).to(card_unit)
                    else:
                        value = value.to(card_unit)
            else:
                # Value is given as a number
                pass

            value = UnitConversion.convert(value, card_unit, int)
            self.card.set_i(SPC_OFFS0 + (SPC_OFFS1 - SPC_OFFS0) * self.index, value)
        
        return_value = self.card.get_i(SPC_OFFS0 + (SPC_OFFS1 - SPC_OFFS0) * self.index)
        # Turn the return value into a quantity
        return_quantity = UnitConversion.to_unit(return_value, return_unit)
        # Save the conversion offset to be able to convert the data to a quantity with the correct unit
        self._conversion_offset = UnitConversion.force_unit(return_value, card_unit)
        return return_quantity
    
    def convert_data(self, data : npt.NDArray, return_unit : pint.Unit = units.mV, averages : int = 1) -> npt.NDArray:
        """
        Converts the data to the correct unit in units of electrical potential
        
        Parameters
        ----------
        data : numpy.ndarray
            The data to be converted
        return_unit : pint.Unit = units.mV
            The unit of the return value
        averages : int = 1
            The number of averages that have been done to the data and should be taken into account to convert the data
            
        Returns
        -------
        numpy.ndarray
            The converted data in units of electrical potential
        """

        max_value = self.card.max_sample_value() * averages
        if self._conversion_offset.check('[]'):
            return_data = (data / max_value - self._conversion_offset) * self._conversion_amp
        else:
            return_data = (data / max_value) * self._conversion_amp - self._conversion_offset
        return_data = UnitConversion.to_unit(return_data, return_unit)
        return return_data
    
    def reconvert_data(self, data : npt.NDArray) -> npt.NDArray:
        """
        Convert data with units back to integer values in units of electrical potential
        
        Parameters
        ----------
        data : numpy.ndarray
            The data to be reconverted
            
        Returns
        -------
        numpy.ndarray
            The reconverted data as integer in mV
        """

        if self._conversion_offset.check('[]'):
            return_data = int((data / self._conversion_amp + self._conversion_offset) * self.card.max_sample_value())
        else:
            return_data = int(((data + self._conversion_offset) / self._conversion_amp) * self.card.max_sample_value())
        return return_data

    def termination(self, value : int) -> None:
        """
        Sets the termination of the analog front-end of the channel of the card (see register `SPC_50OHM0` in the manual)
    
        Parameters
        ----------
        value : int | bool
            The termination of the specific channel
        """

        self.card.set_i(SPC_50OHM0 + (SPC_50OHM1 - SPC_50OHM0) * self.index, int(value))

    def get_termination(self) -> int:
        """
        Gets the termination of the analog front-end of the channel of the card (see register `SPC_50OHM0` in the manual)
            
        Returns
        -------
        int
            The termination of the specific channel
        """

        return self.card.get_i(SPC_50OHM0 + (SPC_50OHM1 - SPC_50OHM0) * self.index)
    
    def coupling(self, value : int = None) -> int:
        """
        Sets the coupling of the analog front-end of the channel of the card (see register `SPC_ACDC0` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling of the specific channel
            
        Returns
        -------
        int
            The coupling of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_ACDC0 + (SPC_ACDC1 - SPC_ACDC0) * self.index, value)
        return self.card.get_i(SPC_ACDC0 + (SPC_ACDC1 - SPC_ACDC0) * self.index)
    
    def coupling_offset_compensation(self, value : int = None) -> int:
        """
        Enables or disables the coupling offset compensation of the analog front-end of the channel of the card (see register `SPC_ACDC_OFFS_COMPENSATION0` in the manual)
    
        Parameters
        ----------
        value : int
            Enables the coupling offset compensation of the specific channel
            
        Returns
        -------
        int
            return if the coupling offset compensation of the specific channel is enabled ("1") or disabled ("0")
        """

        if value is not None:
            self.card.set_i(SPC_ACDC_OFFS_COMPENSATION0 + (SPC_ACDC_OFFS_COMPENSATION1 - SPC_ACDC_OFFS_COMPENSATION0) * self.index, value)
        return self.card.get_i(SPC_ACDC_OFFS_COMPENSATION0 + (SPC_ACDC_OFFS_COMPENSATION1 - SPC_ACDC_OFFS_COMPENSATION0) * self.index)
    
    def filter(self, value : int = None) -> int:
        """
        Sets the filter of the analog front-end of the channel of the card (see register `SPC_FILTER0` in the manual)
    
        Parameters
        ----------
        value : int
            The filter of the specific channel
            
        Returns
        -------
        int
            The filter of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_FILTER0 + (SPC_FILTER1 - SPC_FILTER0) * self.index, value)
        return self.card.get_i(SPC_FILTER0 + (SPC_FILTER1 - SPC_FILTER0) * self.index)
    
    def stop_level(self, value : int = None) -> int:
        """
        Usually the used outputs of the analog generation boards are set to zero level after replay. 
        This is in most cases adequate. In some cases it can be necessary to hold the last sample,
        to output the maximum positive level or maximum negative level after replay. The stoplevel will 
        stay on the defined level until the next output has been made. With this function
        you can define the behavior after replay (see register `SPC_CH0_STOPLEVEL` in the manual)
    
        Parameters
        ----------
        value : int
            The wanted stop behaviour

        Returns
        -------
        int
            The stop behaviour of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_CH0_STOPLEVEL + self.index * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL), value)
        return self.card.get_i(SPC_CH0_STOPLEVEL + self.index * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL))

    def custom_stop(self, value : int = None) -> int:
        """
        Allows to define a 16bit wide custom level per channel for the analog output to enter in pauses. The sample format is 
        exactly the same as during replay, as described in the „sample format“ section.
        When synchronous digital bits are replayed along, the custom level must include these as well and therefore allows to 
        set a custom level for each multi-purpose line separately. (see register `SPC_CH0_CUSTOM_STOP` in the manual)
    
        Parameters
        ----------
        value : int
            The custom stop value

        Returns
        -------
        int
            The custom stop value of the specific channel
        
        TODO: change this to a specific unit?
        """

        if value is not None:
            self.card.set_i(SPC_CH0_CUSTOM_STOP + self.index * (SPC_CH1_CUSTOM_STOP - SPC_CH0_CUSTOM_STOP), value)
        return self.card.get_i(SPC_CH0_CUSTOM_STOP + self.index * (SPC_CH1_CUSTOM_STOP - SPC_CH0_CUSTOM_STOP))
    
    def ch_mask(self) -> int:
        """
        Gets mask for the "or"- or "and"-mask

        Returns
        -------
        int
            The mask for the "or"- or "and"-mask
        """

        return 1 << self.index
    
    def output_load(self, value : pint.Quantity = None) -> pint.Quantity:
        """
        Sets the electrical load of the user system connect the channel of the card. This is important for the correct
        calculation of the output power. Typically, the load would be 50 Ohms, but it can be different.

        Parameters
        ----------
        value : pint.Quantity
            The electrical load connected by the user to the specific channel

        Returns
        -------
        pint.Quantity
            The electrical load connected by the user to the specific channel
        """
        if value is not None:
            self._output_load = value
        return self._output_load
    
    def voltage_conversion(self, value : pint.Quantity) -> pint.Quantity:
        """
        Convert the voltage that is needed at a certain output load to the voltage setting of the card if the load would be 50 Ohm

        Parameters
        ----------
        value : pint.Quantity
            The voltage that is needed at a certain output load

        Returns
        -------
        pint.Quantity
            The corresponding voltage at an output load of 50 Ohm
        """

        # The two at the end is because the value expected by the card is defined for a 50 Ohm load
        if self._output_load == np.inf * units.ohm:
            return value / 2
        return value / (self._output_load / (self._output_load + self._series_impedance)) / 2

    def to_amplitude_fraction(self, value) -> float:
        """
        Convert the voltage, percentage or power to percentage of the full range of the card

        Parameters
        ----------
        value : pint.Quantity | float
            The voltage that should be outputted at a certain output load

        Returns
        -------
        float
            The corresponding fraction of the full range of the card
        """

        if isinstance(value, units.Quantity) and value.check("[power]"):
            # U_pk = U_rms * sqrt(2)
            value = np.sqrt(2 * value.to('mW') * self._output_load) / self._conversion_amp * 100 * units.percent
        elif isinstance(value, units.Quantity) and value.check("[electric_potential]"):
            # value in U_pk
            value = self.voltage_conversion(value) / self._conversion_amp * 100 * units.percent
        value = UnitConversion.convert(value, units.fraction, float, rounding=None)
        return value
    
    def from_amplitude_fraction(self, fraction, return_unit : pint.Quantity = None) -> pint.Quantity:
        """
        Convert the percentage of the full range to voltage, percentage or power

        Parameters
        ----------
        fraction : float
            The percentage of the full range of the card
        return_unit : pint.Quantity
            The unit of the return value

        Returns
        -------
        pint.Quantity
            The corresponding voltage, percentage or power
        """

        return_value = fraction
        if isinstance(return_unit, units.Unit) and (1*return_unit).check("[power]"):
            return_value = (np.power(self._conversion_amp * fraction, 2) / self._output_load / 2).to(return_unit)
            # U_pk = U_rms * sqrt(2)
        elif isinstance(return_unit, units.Unit) and (1*return_unit).check("[electric_potential]"):
            return_value = (self._conversion_amp * fraction / (100 * units.percent)).to(return_unit)
            # value in U_pk
            # value = self.voltage_conversion(value) / self._conversion_amp * 100 * units.percent
        elif isinstance(return_unit, units.Unit) and (1*return_unit).check("[]"):
            return_value = UnitConversion.force_unit(fraction, return_unit)
        return return_value


class Channels:
    """
    a higher-level abstraction of the CardFunctionality class to implement the Card's channel settings
    """
    
    cards : list[Card] = []
    channels : list[Channel] = []
    num_channels : list[int] = []

    def __init__(self, card : Card = None, card_enable : int = None, stack : CardStack = None, stack_enable : list[int] = None) -> None:
        """
        Constructor of the Channels class

        Parameters
        ----------
        card : Card = None
            The card to be used
        card_enable : int = None
            The bitmask to enable specific channels
        stack : CardStack = None
            The card stack to be used
        stack_enable : list[int] = None
            The list of bitmasks to enable specific channels

        Raises
        ------
        SpcmException
            No card or card stack provided
        """

        self.cards = []
        self.channels = []
        self.num_channels = []
        if card is not None:
            self.cards.append(card)
            if card_enable is not None:
                self.channels_enable(enable_list=[card_enable])
            else:
                self.channels_enable(enable_all=True)
        elif stack is not None:
            self.cards = stack.cards
            if stack_enable is not None:
                self.channels_enable(enable_list=stack_enable)
            else:
                self.channels_enable(enable_all=True)
        else:
            raise SpcmException(text="No card or card stack provided")

    def __str__(self) -> str:
        """
        String representation of the Channels class
    
        Returns
        -------
        str
            String representation of the Channels class
        """
        
        return f"Channels()"
    
    __repr__ = __str__

    def __iter__(self) -> "Channels":
        """Define this class as an iterator"""
        return self
    
    def __getitem__(self, index : int) -> Channel:
        """
        This method is called to access the channel by index

        Parameters
        ----------
        index : int
            The index of the channel
        
        Returns
        -------
        Channel
            the channel at the specific index
        """

        if index < 0 or index >= len(self.channels):
            raise IndexError(repr(index))
        return self.channels[index]
    
    _channel_iterator_index = -1
    def __next__(self) -> Channel:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        Channel
            the next available channel
        
        Raises
        ------
        StopIteration
        """
        self._channel_iterator_index += 1
        if self._channel_iterator_index >= len(self.channels):
            self._channel_iterator_index = -1
            raise StopIteration
        return self.channels[self._channel_iterator_index]
    
    def __len__(self) -> int:
        """Returns the number of channels"""
        return len(self.channels)
    
    def write_setup(self) -> None:
        """Write the setup to the card"""
        for card in self.cards:
            card.write_setup()
    
    def channels_enable(self, enable_list : list[int] = None, enable_all : bool = False) -> int:
        """
        Enables or disables the channels of all the available cards (see register `SPC_CHENABLE` in the manual)
    
        Parameters
        ----------
        enable_list : list[int] = None
            A list of channels bitmasks to be enable or disable specific channels
        enable_all : bool = False
            Enable all the channels
        
        Returns
        -------
        int
            A list with items that indicate for each card the number of channels that are enabled, or True to enable all channels.
        """

        self.channels = []
        self.num_channels = []
        num_channels = 0

        if enable_all:
            for card in self.cards:
                num_channels = card.num_channels()
                card.set_i(SPC_CHENABLE, (1 << num_channels) - 1)
                num_channels = card.get_i(SPC_CHCOUNT)
                self.num_channels.append(num_channels)
                for i in range(num_channels):
                    self.channels.append(Channel(i, i, card))
        elif enable_list is not None:
            for enable, card in zip(enable_list, self.cards):
                card.set_i(SPC_CHENABLE, enable)
                num_channels = card.get_i(SPC_CHCOUNT)
                self.num_channels.append(num_channels)
                counter = 0
                for i in range(len(bin(enable))):
                    if (enable >> i) & 1:
                        self.channels.append(Channel(i, counter, card))
                        counter += 1
        return sum(self.num_channels)
        
    # def __getattribute__(self, name):
    #     # print("Calling __getattr__: {}".format(name))
    #     if hasattr(Channel, name):
    #         def wrapper(*args, **kw):
    #             for channel in self.channels:
    #                 getattr(channel, name)(*args, **kw)
    #         return wrapper
    #     else:
    #         return object.__getattribute__(self, name)

    def enable(self, enable : bool) -> None:
        """
        Enables or disables the analog front-end of all channels of the card (see register `SPC_ENABLEOUT` in the manual)
    
        Parameters
        ----------
        enable : bool
            Turn-on (True) or off (False) the spezific channel
        """

        for channel in self.channels:
            channel.enable(enable)
    enable_out = enable
    
    def path(self, value : int) -> None:
        """
        Sets the input path of the analog front-end of all channels of the card (see register `SPC_PATH` in the manual)
    
        Parameters
        ----------
        value : int
            The input path of the specific channel
        """

        for channel in self.channels:
            channel.path(value)
    
    def amp(self, value : int) -> None:
        """
        Sets the output/input range (amplitude) of the analog front-end of all channels of the card in mV (see register `SPC_AMP` in the manual)
    
        Parameters
        ----------
        value : int
            The output range (amplitude) of all channels in millivolts
        """

        for channel in self.channels:
            channel.amp(value)
    
    def offset(self, value : int) -> None:
        """
        Sets the offset of the analog front-end of all channels of the card in mV (see register `SPC_OFFSET` in the manual)
    
        Parameters
        ----------
        value : int
            The offset of all channels in millivolts
        """

        for channel in self.channels:
            channel.offset(value)

    def termination(self, value : int) -> None:
        """
        Sets the termination of the analog front-end of all channels of the card (see register `SPC_50OHM` in the manual)
    
        Parameters
        ----------
        value : int
            The termination of all channels
        """

        for channel in self.channels:
            channel.termination(value)
    
    def digital_termination(self, word_id : int, value : int) -> None:
        """
        Sets the termination of the digital front-end of a specific word (16 channels / bits) of channels of the card (see register `SPC_110OHM0` in the manual)
    
        Parameters
        ----------
        word_id : int
            The ID of the word of channels (e.g. 0 = D15 - D0, 1 = D31 - D16)
        value : bool | int
            The termination of all channels (0 = high-Z, 1 = 110 Ohm)
        """

        for card in self.cards:
            card.set_i(SPC_110OHM0 + word_id * (SPC_110OHM1 - SPC_110OHM0), int(value))
    
    def coupling(self, value : int) -> None:
        """
        Sets the coupling of the analog front-end of all channels of the card (see register `SPC_ACDC` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling of all channels
        """

        for channel in self.channels:
            channel.coupling(value)
    
    def coupling_offset_compensation(self, value : int) -> None:
        """
        Sets the coupling offset compensation of the analog front-end of all channels of the card (see register `SPC_ACDC_OFFS_COMPENSATION` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling offset compensation of all channels
        """

        for channel in self.channels:
            channel.coupling_offset_compensation(value)
    
    def filter(self, value : int) -> None:
        """
        Sets the filter of the analog front-end of all channels of the card (see register `SPC_FILTER` in the manual)
    
        Parameters
        ----------
        value : int
            The filter of all channels
        """

        for channel in self.channels:
            channel.filter(value)
    
    def stop_level(self, value : int) -> None:
        """
        Usually the used outputs of the analog generation boards are set to zero level after replay. 
        This is in most cases adequate. In some cases it can be necessary to hold the last sample,
        to output the maximum positive level or maximum negative level after replay. The stoplevel will 
        stay on the defined level until the next output has been made. With this function
        you can define the behavior after replay (see register `SPC_CH0_STOPLEVEL` in the manual)
    
        Parameters
        ----------
        value : int
            The wanted stop behaviour:
        """

        for channel in self.channels:
            channel.stop_level(value)

    def custom_stop(self, value : int) -> None:
        """
        Allows to define a 16bit wide custom level per channel for the analog output to enter in pauses. The sample format is 
        exactly the same as during replay, as described in the „sample format“ section.
        When synchronous digital bits are replayed along, the custom level must include these as well and therefore allows to 
        set a custom level for each multi-purpose line separately. (see register `SPC_CH0_CUSTOM_STOP` in the manual)
    
        Parameters
        ----------
        value : int
            The custom stop value
        """
    
        for channel in self.channels:
            channel.custom_stop(value)
    
    def output_load(self, value : pint.Quantity) -> None:
        """
        Sets the electrical load of the user system connect the channel of the card. This is important for the correct
        calculation of the output power. Typically, the load would be 50 Ohms, but it can be different.

        Parameters
        ----------
        value : pint.Quantity
            The electrical load connected by the user to the specific channel
        """
        for channel in self.channels:
            channel.output_load(value)
    
    def ch_mask(self) -> int:
        """
        Gets mask for the "or"- or "and"-mask

        Returns
        -------
        int
            The mask for the "or"- or "and"-mask
        """

        return sum([channel.ch_mask() for channel in self.channels])