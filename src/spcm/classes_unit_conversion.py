# -*- coding: utf-8 -*-

from . import units

class UnitConversion:
    """
    Conersion functions for units conversion in the package
    """

    @staticmethod
    def convert(value, base_unit, dtype = int):
        if isinstance(value, units.Quantity):
            return dtype(value.to(base_unit).magnitude)
        return value
    
    @staticmethod
    def to_unit(value, unit):
        if isinstance(value, units.Quantity):
            if isinstance(unit, units.Unit):
                return value.to(unit)
            elif isinstance(unit, units.Quantity):
                return value.to(unit.units)
            else:
                return value.magnitude
        return value
    