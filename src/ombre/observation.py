"""Deal with multiple visits"""
from dataclasses import dataclass
from typing import List
import numpy as np
from .visit import Visit


@dataclass
class Observation(object):
    """Class for storing multiple visits"""

    visits: List[Visit]
    name: str

    def __post_init__(self):
        return

    def __repr__(self):
        return "{} [{} Visits]".format(self.name, len(self))

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, s: int):
        if len(self.visits) == 0:
            raise ValueError("Empty Observation")
        return self.visits[s]

    @staticmethod
    def from_files(fnames: List[str]):
        """Make an observation from files"""
        unique = np.unique([fname.split("/")[-1][:3] for fname in fnames])
        masks = np.asarray(
            [fname.split("/")[-1][:3] == unq for unq in unique for fname in fnames]
        ).reshape((len(unique), len(fnames)))

        visits = [
            Visit.from_files(fnames[mask], forward=direction)
            for mask in masks
            for direction in [True, False]
        ]
        return Observation(visits, name=visits[0].name)

    def fit_transit(self):
        """Fit the transits in all the visits"""
        return

    def _cast_transit_to_visits(self):
        """Puts transit data into the visits"""
        return

    @property
    def time(self):
        return np.hstack([visit.time for visit in obs])

    @property
    def average_lc(self):
        return np.hstack([visit.average_lc for visit in obs])
