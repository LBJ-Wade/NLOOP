"""This module contains utility functions"""


def lazy_property(func):
    """
    turn func into a lazy property

    source: https://stevenloria.com/lazy-properties/"""

    # construct private attribute name by adding an underscore to the front
    attr_name = "_" + str(func.__name__)

    @property
    def lazy_func(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return lazy_func
