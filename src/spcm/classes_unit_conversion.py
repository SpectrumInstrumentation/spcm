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
        if isinstance(value, units.Quantity):
            return_value = value.to(unit)
        else:
            return_value =  value * unit
        return return_value