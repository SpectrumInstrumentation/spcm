# -*- coding: utf-8 -*-

import numpy as np

import pint
from . import units

class UnitConversion:
    """
    Conersion functions for units conversion in the package
    """

    @staticmethod
    def convert(value, base_unit = None, dtype = int, rounding = np.rint):
        """
        Convert a value to a base unit and return the magnitude

        Parameters
        ----------
        value : pint.Quantity
            Value to convert
        base_unit : pint.Unit, optional
            Base unit to convert to, by default None
        dtype : function, optional
            a function to convert the magnitude to a specific type, by default int
        rounding : function, optional
            a function to round the magnitude, by default np.rint

        Returns
        -------
        dtype (int or float)
            magnitude of the value in the base unit
        """
        if isinstance(value, units.Quantity):
            if base_unit is not None:
                base_value = value.to(base_unit)
            else:
                base_value = value.to_base_units()
            if rounding is not None:
                return int(rounding(base_value.magnitude))
            else:
                return dtype(base_value.magnitude)
        return value
    
    @staticmethod
    def to_unit(value, unit : pint.Unit = None):
        """
        Convert a value to a specific unit. If the value has a unit, it will be converted to the specified unit. 
        If the value has no unit, it will be multiplied by the unit that is provided. If the provided unit is a quantity, then this
        function will return the value multiplied by the unit of the quantity.

        Parameters
        ----------
        value : pint.Quantity or int or float
            Value to convert
        unit : pint.Unit or pint.Quantity, optional
            Unit to convert to, by default None

        Returns
        -------
        pint.Quantity or int or float
            Value in the specified unit
        """

        if isinstance(value, units.Quantity):
            if isinstance(unit, units.Unit):
                return value.to(unit)
            elif isinstance(unit, units.Quantity):
                return value.to(unit.units)
            else:
                return value.magnitude
        else:
            if isinstance(unit, units.Unit):
                return value * unit
            elif isinstance(unit, units.Quantity):
                return value * unit.units
            else:
                return value
    
    @staticmethod
    def force_unit(value, unit : pint.Unit):
        """
        Convert a value to a specific unit. If the value has a unit, it will be converted to the specified unit.
        If the value has no unit, it will be multiplied by the unit that is provided. If the provided unit is a quantity, then this
        function will return the value multiplied by the unit of the quantity.

        Parameters
        ----------
        value : pint.Quantity or int or float
            Value to convert
        unit : pint.Unit or pint.Quantity
            Unit to convert to

        Returns
        -------
        pint.Quantity or int or float
            Value in the specified unit
        """
        if isinstance(value, units.Quantity):
            return_value = value.to(unit)
        else:
            return_value =  value * unit
        return return_value