"""
Minimal ONVIF discovery/client helpers.
"""

from .discovery import discover_onvif, OnvifDiscoveryResult
from .client import (
    OnvifClient,
    OnvifAuthError,
    OnvifError,
    OnvifHttpError,
    try_onvif_zeep_stream,
)

__all__ = [
    "discover_onvif",
    "OnvifDiscoveryResult",
    "OnvifClient",
    "OnvifAuthError",
    "OnvifError",
    "OnvifHttpError",
    "try_onvif_zeep_stream",
]
