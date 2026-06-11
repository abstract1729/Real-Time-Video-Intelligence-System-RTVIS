"""
base_event.py

Abstract base class for all event modules.

Every event detector should inherit from BaseEvent and implement
the process() method.

Author: RTVIS
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseEvent(ABC):
    """
    Abstract interface for event modules.

    All event detectors should inherit from this class and implement
    the process() method.
    """

    @abstractmethod
    def process(self, track: Dict) -> List[Dict]:
        """
        Process a single track and generate events.

        Parameters
        ----------
        track : Dict
            Track information obtained from TrackManager.

        Returns
        -------
        List[Dict]
            List of event dictionaries.

            Example:
            [
                {
                    "event_type": "line_crossing",
                    "track_id": 5,
                    "timestamp": 1718112.2
                }
            ]
        """
        pass