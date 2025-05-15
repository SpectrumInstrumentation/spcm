# -*- coding: utf-8 -*-

import ctypes
import platform
from typing import Union

from spcm_core import spcm_dwGetParam_i64, spcm_hOpen, spcm_vClose
from spcm_core.constants import *

from .classes_device import Device

from .classes_error_exception import SpcmException

class Card(Device):
    """
    a high-level class to control Spectrum Instrumentation cards

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    """

    _std_device_identifier : str = "/dev/spcm{}"
    _max_cards : int = 64
    
    _function_type : int = 0
    _card_type : int = 0
    _max_sample_value : int = 0
    _features : int = 0
    _ext_features : int = 0
    _demo_card : bool = False

    def __enter__(self) -> 'Card':
        """
        Context manager entry function

        Returns
        -------
        Card
            The card object
        
        Raises
        ------
        SpcmException
        """
        return super().__enter__()

    def open(self, device_identifier : str = None) -> 'Card':
        """
        Open a connection to the card

        Parameters
        ----------
        device_identifier : str = ""
            The device identifier of the card that needs to be opened

        Returns
        -------
        Card
            The card object

        Raises
        ------
        SpcmException
        """

        if device_identifier is not None:
            return super().open(device_identifier=device_identifier)
        
        super().open()

        # keyword arguments
        card_type = self._kwargs.get("card_type", 0)
        serial_number = self._kwargs.get("serial_number", 0)

        if self.device_identifier == "":
            # No device identifier was given, so we need to find the first card
            self._handle = self.find(card_type=card_type, serial_number=serial_number)
            if not self._handle:
                if card_type:
                    raise SpcmException(text="No card found of right type")
                elif serial_number:
                    raise SpcmException(text="No card found with serial number: {}".format(serial_number))
            else:
                self._closed = False
        elif self._handle:
            if card_type != 0 and self.function_type() != card_type:
                raise SpcmException(text="The card with the given device identifier is not the correct type")
            elif serial_number != 0 and self.sn() != serial_number:
                raise SpcmException(text="The card with the given device identifier does not have the correct serial number")
        
        # Check python, driver and kernel version
        if self._verbose:
            print("Python version: {} on {}".format (platform.python_version(), platform.system()))
            print("Driver version: {major}.{minor}.{build}".format(**self.drv_version()))
            print("Kernel version: {major}.{minor}.{build}".format(**self.kernel_version()))
            if self._handle:
                print("Found '{}': {} sn {:05d}".format(self.device_identifier, self.product_name(), self.sn()))

        # Get the function type of the card
        self._function_type = self.get_i(SPC_FNCTYPE)
        self._card_type = self.get_i(SPC_PCITYP)
        self._features = self.get_i(SPC_PCIFEATURES)
        self._ext_features = self.get_i(SPC_PCIEXTFEATURES)
        self._max_sample_value = self.get_i(SPC_MIINST_MAXADCVALUE)
        self._demo_card = bool(self.get_i(SPC_MIINST_ISDEMOCARD))
        
        return self
    
    def __str__(self) -> str:
        """
        String representation of the card

        Returns
        -------
        str
            String representation of the card
        """
        return "Card: {} sn {:05d}".format(self.product_name(), self.sn())
    __repr__ = __str__
    
    def find(self, card_type : int = 0, serial_number : int = 0) -> Union[bool, int]:
        """
        Find first card that is connected to the computer, with either the given card type or serial number

        Parameters
        ----------
        card_type : int = 0
            The function type of the card that needs to be found
        serial_number : int = 0
            The serial number of the card that needs to be found
        
        Returns
        -------
        Union[bool, int]
            False if no card is found, otherwise the handle of the card

        """
        for nr in range(self._max_cards):
            device_identifier = self._std_device_identifier.format(nr)
            handle = spcm_hOpen(ctypes.create_string_buffer(bytes(device_identifier, 'utf-8')))
            if handle:
                self.device_identifier = device_identifier
                return_value = ctypes.c_int64()
                spcm_dwGetParam_i64(handle, SPC_FNCTYPE, ctypes.byref(return_value))
                function_type = return_value.value
                spcm_dwGetParam_i64(handle, SPC_PCISERIALNO, ctypes.byref(return_value))
                sn = return_value.value
                if card_type != 0 and (card_type & function_type) == function_type:
                    return handle
                elif sn != 0 and sn == serial_number:
                    return handle
                elif serial_number == 0 and card_type == 0:
                    return handle
                spcm_vClose(handle)
        return False

                
    # High-level parameter functions, that use the low-level get and set function
    def status(self) -> int:
        """
        Get the status of the card (see register `SPC_M2STATUS` in the manual)
    
        Returns
        -------
        int
            The status of the card
        """

        return self.get_i(SPC_M2STATUS)
    
    def card_type(self) -> int:
        """
        Get the card type of the card (see register `SPC_PCITYP` in the manual)
    
        Returns
        -------
        int
            The card type of the card
        """

        return self._card_type
    
    def series(self) -> int:
        """
        Get the series of the card (see register `SPC_PCITYP` and `TYP_SERIESMASK` in the manual)
    
        Returns
        -------
        int
            The series of the card
        """

        return self.card_type() & TYP_SERIESMASK
    
    def family(self) -> int:
        """
        Get the family of the card (see register `SPC_PCITYP` and `TYP_FAMILYMASK` in the manual)
    
        Returns
        -------
        int
            The family of the card
        """

        return (self.card_type() & TYP_FAMILYMASK) >> 8

    def function_type(self) -> int:
        """
        Gives information about what type of card it is. (see register `SPC_FNCTYPE` in the manual)
    
        Returns
        -------
        int
            The function type of the card

            * SPCM_TYPE_AI = 1h - Analog input card (analog acquisition; the M2i.4028 and M2i.4038 also return this value)
            * SPCM_TYPE_AO = 2h - Analog output card (arbitrary waveform generators)
            * SPCM_TYPE_DI = 4h - Digital input card (logic analyzer card)
            * SPCM_TYPE_DO = 8h - Digital output card (pattern generators)
            * SPCM_TYPE_DIO = 10h - Digital I/O (input/output) card, where the direction is software selectable.
        """

        return self._function_type

    def features(self) -> int:
        """
        Get the features of the card (see register `SPC_PCIFEATURES` in the manual)
    
        Returns
        -------
        int
            The features of the card
        """

        return self._features
    
    def ext_features(self) -> int:
        """
        Get the extended features of the card (see register `SPC_PCIEXTFEATURES` in the manual)
    
        Returns
        -------
        int
            The extended features of the card
        """

        return self._ext_features
    
    def starhub_card(self) -> bool:
        """
        Check if the card is a starhub card (see register `SPC_PCIFEATURES` in the manual)
    
        Returns
        -------
        bool
            True if the card is the card that carriers a starhub, False otherwise
        """

        return bool(self._features & SPCM_FEAT_STARHUBXX_MASK)
    
    def num_modules(self) -> int:
        """
        Get the number of modules of the card (see register `SPC_MIINST_MODULES` in the manual)
    
        Returns
        -------
        int
            The number of modules of the card
        """

        return self.get_i(SPC_MIINST_MODULES)

    def channels_per_module(self) -> int:
        """
        Get the number of channels per module of the card (see register `SPC_MIINST_CHPERMODULE` in the manual)
    
        Returns
        -------
        int
            The number of channels per module of the card
        """

        return self.get_i(SPC_MIINST_CHPERMODULE)
    
    def num_channels(self) -> int:
        """
        Get the number of channels of the card (= SPC_MIINST_MODULES * SPC_MIINST_CHPERMODULE)
    
        Returns
        -------
        int
            The number of channels of the card
        """

        return self.num_modules() * self.channels_per_module()
    
    def is_demo_card(self) -> bool:
        """
        Check if the card is a demo card (see register `SPC_MIINST_ISDEMOCARD` in the manual)
    
        Returns
        -------
        bool
            True if the card is a demo card, False otherwise
        """

        return self._demo_card

    def card_mode(self, card_mode : int = None) -> int:
        """
        Set the card mode of the connected card (see register `SPC_CARDMODE` in the manual)
        
        Parameters
        ----------
        card_mode : int
            the mode that the card needs to operate in

        Returns
        -------
        int
            the mode that the card operates in
        """
        
        if card_mode is not None:
            self.set_i(SPC_CARDMODE, card_mode)
        return self.get_i(SPC_CARDMODE)

    def product_name(self) -> str:
        """
        Get the product name of the card (see register `SPC_PCITYP` in the manual)
    
        Returns
        -------
        str
            The product name of the connected card (e.g. M4i.6631-x8)
        """

        return self.get_str(SPC_PCITYP)
    
    def sn(self) -> int:
        """
        Get the serial number of a product (see register `SPC_PCISERIALNO` in the manual)
    
        Returns
        -------
        int
            The serial number of the connected card (e.g. 12345)
        """

        return self.get_i(SPC_PCISERIALNO)

    def active_channels(self) -> int:
        """
        Get the number of channels of the card (see register `SPC_CHCOUNT` in the manual)
    
        Returns
        -------
        int
            The number of channels of the card
        """

        return self.get_i(SPC_CHCOUNT)

    def bits_per_sample(self) -> int:
        """
        Get the number of bits per sample of the card (see register `SPC_MIINST_BITSPERSAMPLE` in the manual)
    
        Returns
        -------
        int
            The number of bits per sample of the card
        """

        return self.get_i(SPC_MIINST_BITSPERSAMPLE)
    
    def bytes_per_sample(self) -> int:
        """
        Get the number of bytes per sample

        Returns
        -------
        int
            number of bytes per sample
        """
        return self.get_i(SPC_MIINST_BYTESPERSAMPLE)
    
    def max_sample_value(self) -> int:
        """
        Get the maximum ADC value of the card (see register `SPC_MIINST_MAXADCVALUE` in the manual)

        Returns
        -------
        int
            The maximum ADC value of the card
        """

        return self._max_sample_value
    
    def loops(self, loops : int = None) -> int:
        """
        Set the number of times the memory is replayed. If set to zero the generation will run continuously until it is 
        stopped by the user.  (see register `SPC_LOOPS` in the manual)
        
        Parameters
        ----------
        loops : int
            the number of loops that the card should perform
        """

        if loops is not None:
            self.set_i(SPC_LOOPS, loops)
        return self.get_i(SPC_LOOPS)