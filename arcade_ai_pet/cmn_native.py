"""Trusted CMN C runtime. No firmware writes or generated C input allowed."""
import ctypes as C
from . import cmn_assets
from .decision_native import DecisionNativeModel, SOURCES


class CmnNativeModel(DecisionNativeModel):
    asset_profile = cmn_assets
    sources = (*SOURCES, 'cmn_features.c', 'cmn_features.h')
    feature_prefix = 'pet_cmn'

    def __init__(self, directory):
        super().__init__(directory)
        self.lib.pet_cmn_center.argtypes = [C.c_void_p, C.c_size_t]
