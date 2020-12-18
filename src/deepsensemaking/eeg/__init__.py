#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
deepsensemaking (dsm) eeg auxiliary tools


"""

from deepsensemaking.eeg import mne



from deepsensemaking.dicts import str_dict,print_dict
from deepsensemaking.bids  import get_bids_prop


from collections import OrderedDict



def convert_ZZ(dfZZ,ifSetup="peaks/data/setup.json",verbose=3,):
    """Convert Brain Vision Analizer PEAK export from ZZ to something
    more like my pandas exports

    """

    # Add relevant columns
    cols0 = [
        "evoked0",
        "quest0",
        "cond0",
        "chan0",
        "tmin0",
        "tmax0",
        "mode0",
        "chanX",
        "latX",
        "valX",
        "SUB",
        "SES",
        "TASK",
        "RUN",
        "CHAN_BUND",
    ]
    for col0 in cols0:
        dfZZ[col0] = None


    # Load study setup
    with open(ifSetup) as fhSetup: setup = OrderedDict(json.load(fhSetup))


    # Projects swapping keys from `str` to `int`
    cond_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["cond"].items():
            cond_swaps[int(key0)] = val0

    tmin_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["time"].items():
            tmin_swaps[int(key0)] = val0[0]

    tmax_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["time"].items():
            tmax_swaps[int(key0)] = val0[1]

    if verbose > 2:
        print_dict(cond_swaps,"cond_swaps")
        print_dict(tmin_swaps,"tmin_swaps")
        print_dict(tmax_swaps,"tmax_swaps")


    # Provide dictionary to translate from channels to channel bundles
    chan_swaps = OrderedDict()
    for key0,val0 in setup["chans"]["bund0"].items():
        for item0 in val0:
            chan_swaps[item0] = key0

    if verbose > 2:
        print_dict(chan_swaps,"chan_swaps")


    # TRANSLATE
    dfZZ["evoked0"]   = "evokedZ" # This column typically indicates preprocessing stage
    dfZZ["quest0"]    = "word_set" # Here we have only one quest (additionally `word_len` is present in MNE data)
    dfZZ["cond0"]     = dfZZ["COND_NUM"].map(cond_swaps) # POS/m/f derrived conditions
    # dfZZ["chan0"]     = dfZZ["ELEC"].map(chan_swaps) # Translation of channel names to channel bundles
    dfZZ["chan0"]     = dfZZ["ELEC"]
    dfZZ["tmin0"]     = dfZZ["TIME_WINDOW"].map(tmin_swaps)
    dfZZ["tmax0"]     = dfZZ["TIME_WINDOW"].map(tmax_swaps)
    dfZZ["mode0"]     = "pos" # Not so important now but `pos` turns out to be the right option
    # dfZZ["chanX"]     = dfZZ["ELEC"].map(chan_swaps) # same as above
    dfZZ["chanX"]     = dfZZ["ELEC"]
    dfZZ["latX"]      = np.nan # No peak latency data was provided
    dfZZ["valX"]      = dfZZ["PEAK"] # Peak aplitude
    dfZZ["SUB"]       = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="sub"))  # Extract subject code from file name
    dfZZ["SES"]       = "eeg001" # Session code
    dfZZ["TASK"]      = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="task")) # Extract task code from file name
    dfZZ["RUN"]       = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="run"))  # Extract run number from file name

    dfZZ["CHAN_BUND"] = np.nan # This is NaN for bundles that are set in `chan0` and `chanX` columns
    dfZZ["CHAN_BUND"] = dfZZ["ELEC"].map(chan_swaps)


    return dfZZ




def chan_bund_mean(df3):
    pass
