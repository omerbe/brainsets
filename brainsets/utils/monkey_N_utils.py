import numpy as np
import pandas as pd

from temporaldata import ArrayDict, IrregularTimeSeries

from brainsets.descriptions import SubjectDescription
from brainsets.taxonomy import (
    RecordingTech,
    Sex,
    Species,
)

def monkey_N_subject(nwbfile):
    r"""DANDI has requirements for metadata included in `subject`. This includes:
    subject_id: A subject identifier must be provided.
    species: either a latin binomial or NCBI taxonomic identifier.
    sex: must be "M", "F", "O" (other), or "U" (unknown).
    date_of_birth or age: this does not appear to be enforced, so will be skipped.
    """

    return SubjectDescription(
        id="n",
        species=Species.from_string("MACACA_MULATTA"), #rhesus macaque
        sex=Sex.from_string("M"), 
    )