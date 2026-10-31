"""A natively interpretable linear model for CYP time-dependent inhibition.

    from glassbox import GlassBox, FeatureSet
    m = GlassBox.load('models/CYP3A4.json')
    print(GlassBox.summary(m.explain(fs, row)))
"""
from .model import GlassBox, FeatureSet          # noqa: F401
from . import data, features, select             # noqa: F401

__version__ = '1.0.0'
__all__ = ['GlassBox', 'FeatureSet', 'data', 'features', 'select']
