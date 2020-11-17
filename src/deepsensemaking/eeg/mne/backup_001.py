#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
deepsensemaking (dsm) eeg/mne auxiliary tools


"""

import os
import sys
import glob
import pathlib

import json
import numpy  as np
import pandas as pd
import hickle as hkl

import warnings
import logging

import inspect

"""
checkup:
- https://mne.tools/stable/auto_examples/decoding/plot_ems_filtering.html#sphx-glr-auto-examples-decoding-plot-ems-filtering-py

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import humanfriendly as hf

import time
import datetime as dt
from pytz import timezone as tz
loc_tz = tz("Europe/Berlin")

from pprint import pprint  as pp
from pprint import pformat as pf

from copy import deepcopy as dc

from collections import OrderedDict
from collections import UserList
from collections import UserDict

from   deepsensemaking.bids  import get_bids_prop
from   deepsensemaking.dicts import str_dict,print_dict


import mne
mne.set_log_level("WARNING")
mne.set_log_level("INFO")


# inspect.currentframe().f_back.f_code


def whoami():
    # frame = inspect.currentframe()
    # frame = inspect.currentframe().f_back.f_code
    frame = inspect.currentframe().f_back
    return inspect.getframeinfo(frame).function


def foo():
    print(whoami())


def get_int_input(prompt,valmin,valmax):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("That was not an integer!")
            continue

        if   value < valmin:
            print("That was too low!")
            continue
        elif value > valmax:
            print("That was too high!")
            continue
        else:
            break

    print("got: "+str(value))
    return value


space0 = [ ""   , "- "   , "  - "   , "    - "   , ]
space1 = [ "\n" , "\n  " , "\n    " , "\n      " , ]


class BatchMNE:
    """MNE batch job class"""


    def __init__(
            self,
            objName     = "DS",
            sourceDir   = "rawdata",
            targetDir   = "derivatives",
            globSuffix  = "sub-*/ses-*/eeg/sub-*.vhdr",
            setupFile   = "setup.json",
            stimuliFile = "stimuli.csv",
            verbose     = 0,
    ):

        ## Ensure that paths are of type pathlib.Path
        sourceDir   = pathlib.Path(sourceDir)
        targetDir   = pathlib.Path(targetDir)
        globSuffix  = pathlib.Path(globSuffix)
        setupFile   = pathlib.Path(setupFile)
        stimuliFile = pathlib.Path(stimuliFile)

        ## Basic assertions
        assert isinstance(objName,(str,)),"PROBLEM: expected objName to be of type \"string\", got {}".format(str(type(objName)))
        assert isinstance(verbose,(int,float,complex,)),"PROBLEM: expected verbose to be a number, got {}".format(str(type(verbose)))
        assert sourceDir.   exists()  ,"PROBLEM: provided sourceDir"   + "path does not exist"
        assert setupFile.   exists()  ,"PROBLEM: provided setupFile"   + "path does not exist"
        assert stimuliFile. exists()  ,"PROBLEM: provided stimuliFile" + "path does not exist"
        assert sourceDir.   is_dir()  ,"PROBLEM: provided sourceDir"   + "path is not a directory"
        assert setupFile.   is_file() ,"PROBLEM: provided setupFile"   + "path is not a file"
        assert stimuliFile. is_file() ,"PROBLEM: provided stimuliFile" + "path is not a file"

        ## Basic class attributes
        self.objName     = objName
        self.INSP        = [objName]
        self.sourceDir   = sourceDir
        self.targetDir   = targetDir
        self.loggerDir   = self.targetDir/"logs"
        self.globSuffix  = globSuffix
        self.globPattern = self.sourceDir/self.globSuffix
        self.setupFile   = setupFile
        self.stimuliFile = stimuliFile
        self.verbose     = verbose

        ## Prepare target directories
        os.makedirs(self.targetDir,mode=0o700,exist_ok=True,)
        os.makedirs(self.loggerDir,mode=0o700,exist_ok=True,)

        ## Setup the logger
        self.logging = logging
        self.logger  = logging.getLogger(__name__)
        """
        self.logger.setLevel(logging.CRITICAL) # 50
        self.logger.setLevel(logging.ERROR)    # 40
        self.logger.setLevel(logging.WARNING)  # 30
        self.logger.setLevel(logging.INFO)     # 20
        self.logger.setLevel(logging.DEBUG)    # 10
        self.logger.setLevel(logging.NOTSET)   # 00
        """
        self.logger.setLevel(logging.DEBUG)    # 10
        self.handler0 = logging.StreamHandler()
        self.handler0.setLevel(logging.INFO)
        self.handler0.setFormatter(
            logging.Formatter(" ".join([
                # "%(asctime)s",
                # "%(name)s",
                "%(levelname).1s:",
                # "%(module)s",
                "%(funcName)-16s ",
                "%(message)s",
            ]),
            datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        fn0 = self.loggerDir/(dt.datetime.now(loc_tz).strftime("%Y%m%d_%H%M%S_%f")[:-3]+".log")
        self.handler1 = logging.FileHandler(fn0)
        self.handler1.setLevel(logging.DEBUG)
        self.handler1.setFormatter(
            logging.Formatter(" ".join([
                "%(asctime)s",
                # "%(name)s",
                "%(levelname).1s:",
                # "%(module)s",
                "%(funcName)-16s ",
                "%(message)s",
            ]),
            datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        self.logger.addHandler(self.handler0)
        self.logger.addHandler(self.handler1)

        ## Also attach MNE logger???
        temp_attach_MNE_logger = False
        temp_attach_MNE_logger = True
        if temp_attach_MNE_logger:
            for handler in mne.utils.logger.handlers[:]: mne.utils.logger.removeHandler(handler)
            mne.utils.logger.setLevel(logging.DEBUG)
            mne.utils.logger.addHandler(self.handler0)
            mne.utils.logger.addHandler(self.handler1)

        self.logger.info (space0[0]+"logging to: "+str(fn0))
        self.logger.info (space0[0]+"handler0 level: "+str(logging.getLevelName(self.handler0)))
        self.logger.info (space0[0]+"handler1 level: "+str(logging.getLevelName(self.handler1)))
        self.logger.info (space0[0]+"MNE version: " + str(mne.__version__))
        self.logger.info (space0[0]+self.objName +" (BatchMNE) was CASTED (NICE)!!!")

        self.inputPaths = self.InputPaths(BATCH=self,INSP=self.INSP,objName="inputPaths")
        self.dataBase   = self.DataBase  (BATCH=self,INSP=self.INSP,objName="dataBase"  )


    def info(self):
        self.logger.info(space0[0]+"RUNNING: {}.{}".format(
            ".".join(self.INSP),
            str(whoami()),
        ))
        self.logger.info(self.__str__())


    def __repr__(self):
        return self.__str__()


    def __str__(self):
        out_str  = ""
        out_str += space0[1]+"{} is a {} object  properties such as:".format(
            self.objName,
            type(self).__name__,
        )
        out_str += "\n"
        out_str += space1[1]+self.objName+".objName     = "+repr(str(self.objName    ))
        out_str += space1[1]+self.objName+".sourceDir   = "+repr(str(self.sourceDir  ))
        out_str += space1[1]+self.objName+".targetDir   = "+repr(str(self.targetDir  ))
        out_str += space1[1]+self.objName+".loggerDir   = "+repr(str(self.loggerDir  ))
        out_str += space1[1]+self.objName+".globSuffix  = "+repr(str(self.globSuffix ))
        out_str += space1[1]+self.objName+".globPattern = "+repr(str(self.globPattern))
        out_str += space1[1]+self.objName+".setupFile   = "+repr(str(self.setupFile  ))
        out_str += space1[1]+self.objName+".stimuliFile = "+repr(str(self.stimuliFile))
        out_str += space1[1]+self.objName+".verbose     = "+str(self.verbose    )
        out_str += space1[1]+self.objName+".logger      : "+str(self.logger     )
        out_str += space1[1]+self.objName+".inputPaths  : "+"contains {} items".format(len(self.inputPaths))
        out_str += space1[1]+self.objName+".dataBase    : "+"contains {} items".format(len(self.dataBase))
        out_str += "\n"
        return out_str


    class InputPaths(UserList):


        def __init__(self,BATCH,INSP,objName):
            UserList.__init__(self)
            self.objName = objName
            self.INSP = [item for item in INSP]+[objName]
            # self.DPTH = len(self.INSP)-2
            self.BATCH   = BATCH


        def glob(self):
            self.BATCH.logger.info(space0[0]+"RUNNING: {}.{}".format(
                ".".join(self.INSP),
                str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"globbing for input data paths...")
            self.BATCH.logger.info(space0[1]+"pattern: {}".format(repr(str(self.BATCH.globPattern))))
            self.data = glob.glob(str(self.BATCH.globPattern))
            self.BATCH.logger.info(space0[1]+"got {} items".format(len(self.data)))


        def select(self,if_selector_path,mode="keep"):
            self.BATCH.logger.info(space0[0]+"RUNNING: {}.{}".format(
                ".".join(self.INSP),
                str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"selecting input paths")

            if_selector_path = pathlib.Path(if_selector_path)
            assert if_selector_path.exists(), "PROBLEM: Provided selector file was not found!"
            self.BATCH.logger.info(space0[1]+"if_selector_path: {}".format(repr(str(if_selector_path))))
            self.BATCH.logger.info(space0[1]+"mode: {}".format(repr(str(mode))))
            selector_str_list = list()
            with open(if_selector_path) as fh0:
                for line in fh0:
                    line = line.strip()
                    if line:
                        selector_str_list.append(line.strip())

            if selector_str_list:
                self.BATCH.logger.info(space0[1]+"selector_str_list contains {} items:".format(len(selector_str_list)))
                for item in selector_str_list:
                    self.BATCH.logger.info(space0[2]+repr(item))

                if mode=="keep":
                    self.data = [ item for item in self.data if     any( item for if_selector_path in selector_str_list if if_selector_path in item ) ]
                else:
                    self.data = [ item for item in self.data if not any( item for if_selector_path in selector_str_list if if_selector_path in item ) ]

            else:
                self.BATCH.logger.warning(space0[1]+"selector_str_list contains {} items (??? WTF ???)".format(
                    len(selector_str_list),
                ))

            self.BATCH.logger.info(space0[1]+"{} ({}) is returning {} items".format(
                ".".join(self.INSP),
                type(self).__name__,
                len(self.data),
            ))


        def length(self):
            return self.data.__len__()


        def length_info(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(
                space0[1]+"{} contains {} items:".format(
                    ".".join(self.INSP),
                    self.length(),
            ))


        def info(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            out_str  = ""
            out_str += space0[1]+"{} contains {} items:".format(
                    ".".join(self.INSP),
                    self.length(),
            )
            out_str += "\n"
            if self.data:
                for item in self.data:
                    out_str += space1[1]+item

                out_str += "\n"
                self.BATCH.logger.info(out_str)

            else:
                out_str = space1[2]+"No input paths to display!"
                out_str += "\n"
                self.BATCH.logger.warning(out_str)



    class DataBase(UserList):


        def __init__(self,BATCH,INSP,objName,):
            UserList.__init__(self)
            self.objName = objName
            self.INSP = [item for item in INSP]+[objName]
            self.BATCH   = BATCH
            self.setup   = self.Setup(self.BATCH,self.INSP,objName="setup")


        def get_paths(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"getting data paths...")
            self.data = list()
            for idx,item in enumerate(self.BATCH.inputPaths):
                self.data.append(
                    self.DataSet(
                        BATCH = self.BATCH,
                        INSP  = self.INSP ,
                        item  = item,
                        idx   = idx,
                    )
                )
            self.BATCH.logger.info(space0[1]+"got {} data paths".format(
                len(self.data),
            ))


        def read_ALL_raw(self,raw0="raw0",preload=True,verbose=None):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"getting data for ALL paths")
            for idx,item in enumerate(self.data):
                self.BATCH.logger.info(space0[1]+"reading RAW data: {}".format(item))
                self.data[idx].read_raw_data(raw0=raw0,preload=preload,verbose=verbose)


        def read_ALL_hkl(self,raw0="raw0",preload=True,verbose=None):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"getting data for ALL paths")
            for idx,item in enumerate(self.data):
                self.BATCH.logger.info(space0[1]+"reading HKL data: {}".format(item))
                self.data[idx].read_hkl_data()


        def BATCH_001_001_ica(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            for idx,item in enumerate(self.data):
                if not item.locs.of_done.is_file():
                    self.BATCH.logger.info(space0[1]+"PROCESSING: [{}] {}".format(idx,item,))
                    self.data[idx].BATCH_001_001_ica()


        def info(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            out_str  = ""
            out_str += space0[1]+"{} contains {} items:".format(
                    ".".join(self.INSP),
                    self.length(),
            )
            out_str += "\n"
            for idx,item in enumerate(self.data):
                temp_status = "[    ]" if item.locs.of_done.is_file() else "[TODO]"
                #out_str += " "*2+str(item.locs.of_stem)+": "+str(len(item.data.keys()))+"\n"
                out_str += space1[2]+"{:>{}d}: {} {}".format(
                    idx,
                    len(str(len(self.data)-1)),
                    temp_status,
                    item,
                )
                if len(item.data.keys()) > 0:
                    out_str += space1[2]+"{} {}".format(
                        " "*len(str(len(self.data)))+" "*8,
                        repr(list(item.data.keys())),
                    )

            out_str += "\n"
            self.BATCH.logger.info(out_str)


        def length(self):
            return self.__len__()


        def length_info(self):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(
                space0[1]+"{} contains {} items:".format(
                    ".".join(self.INSP),
                    self.length(),
            ))


        class Setup(UserDict):


            def __init__(self,BATCH,INSP,objName):
                UserDict.__init__(self)
                self.objName = objName
                self.INSP    = [item for item in INSP]+[objName]
                self.BATCH   = BATCH


            def read(self):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))

                self.BATCH.logger.info(space0[1]+"getting SETUP DATA [1/2]...")
                fn0 = self.BATCH.setupFile
                self.BATCH.logger.info(space0[2]+"setupFile: {}".format(repr(str(fn0))))
                with open(fn0) as fh0: self.data = OrderedDict(json.load(fh0))

                self.BATCH.logger.info(space0[1]+"getting STIMULI DATA [2/2]...")
                fn1 = self.BATCH.stimuliFile
                self.BATCH.logger.info(space0[2]+"stimuliFile: {}".format(repr(str(fn1))))
                self.data["events"]["dgn0"] = pd.read_csv(fn1)

                self.BATCH.logger.info(space0[1]+"All good!")


            def info(self):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                out_str  = ""
                out_str += space0[1]+"{} is a dict-like object containing:".format(
                    ".".join(self.INSP),
                )
                out_str += "\n"
                out_str += str_dict(
                    self.data,
                    "   {}".format(
                        ".".join(self.INSP),
                    )
                )
                out_str += "\n"
                self.BATCH.logger.info(out_str)


            def stimuli_info(self):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                key0 = "events"
                key1 = "dgn0"
                out_str  = ""
                out_str += space0[1]+"{}[{}][{}] is a pandas DataFrame:".format(
                    ".".join(self.INSP),
                        str(repr(key0)),
                        str(repr(key1)),
                )
                out_str += "\n\n" # TODO FIXME add four spaces to the left of each line
                with pd.option_context("display.max_colwidth"         ,  200):
                    with pd.option_context("display.width"            , 3500):
                        with pd.option_context("display.max_rows"     ,   45):
                            with pd.option_context("display.min_rows" ,   45):
                                out_str += pf(self[key0][key1])

                self.BATCH.logger.info(out_str)


        class DataSet():


            def __init__(self,BATCH,INSP,item,idx):
                self.BATCH    = BATCH
                self.INSP     = [item for item in INSP]
                self.INSP[-1] = self.INSP[-1]+"[{}]".format(idx)
                self.locs     = self.Locs(self.BATCH,self.INSP,item,objName="locs")
                self.data     = OrderedDict()
                os.makedirs(self.locs.od_path,mode=0o700,exist_ok=True,)


            def __repr__(self):
                return self.__str__()


            def __str__(self):
                return str(self.locs.of_stem) + " ("+str(len(self.get_keys()))+")"


            def __len__(self):
                return len(self.data.keys())


            def get_keys(self):
                return list(self.data.keys())


            def info(self):
                out_str  = ""
                out_str += space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                )
                out_str += "\n"
                out_str += space1[1]+self.__str__()
                out_str += space1[1]+str(self.get_keys())
                out_str += "\n"
                self.BATCH.logger.info(out_str)


            class Locs:


                def __init__(self,BATCH,INSP,item,objName):
                    self.objName       = objName
                    self.INSP          = [item for item in INSP]+[objName]
                    self.BATCH         = BATCH
                    self.if_path       = pathlib.Path(item)
                    self.of_path       = BATCH.targetDir / self.if_path.relative_to(BATCH.sourceDir)
                    self.od_path       = self.of_path.parents[0]
                    self.of_stem       = pathlib.Path(self.of_path.stem.split('.')[0])
                    self.of_base       = self.od_path / self.of_stem
                    self.of_BAD_chans  = self.of_base.with_suffix(".BAD_chans")
                    self.of_BAD_spans  = self.of_base.with_suffix(".BAD_spans.csv")
                    self.of_BAD_epochs = self.of_base.with_suffix(".BAD_epochs")
                    self.of_BAD_comps  = self.of_base.with_suffix(".BAD_comps")
                    self.of_rand       = self.of_base.with_suffix(".rand")
                    self.of_done       = self.of_base.with_suffix(".gzip.hkl")


                def __repr__(self):
                    return self.__str__()


                def __str__(self):
                    out_str  = ""
                    out_str += space1[1]+"       if_path = {}".format(repr(str( self.if_path       )))
                    out_str += space1[1]+"       of_path = {}".format(repr(str( self.of_path       )))
                    out_str += space1[1]+"       od_path = {}".format(repr(str( self.od_path       )))
                    out_str += space1[1]+"       of_stem = {}".format(repr(str( self.of_stem       )))
                    out_str += space1[1]+"       of_base = {}".format(repr(str( self.of_base       )))
                    out_str += space1[1]+"  of_BAD_chans = {}".format(repr(str( self.of_BAD_chans  )))
                    out_str += space1[1]+"  of_BAD_spans = {}".format(repr(str( self.of_BAD_spans  )))
                    out_str += space1[1]+" of_BAD_epochs = {}".format(repr(str( self.of_BAD_epochs )))
                    out_str += space1[1]+"  of_BAD_comps = {}".format(repr(str( self.of_BAD_comps  )))
                    out_str += space1[1]+"       of_rand = {}".format(repr(str( self.of_rand       )))
                    out_str += space1[1]+"       of_done = {}".format(repr(str( self.of_done       )))

                    return out_str


                def info(self):
                    self.BATCH.logger.info(
                        space0[0]+"RUNNING: {}.{}".format(
                            ".".join(self.INSP),
                            str(whoami()),
                    ))
                    out_str  = ""
                    out_str += space0[1]+"{} contains (i/o) patchs such as:".format(
                        ".".join(self.INSP),
                    )
                    out_str += "\n"
                    out_str += self.__str__()
                    out_str += "\n"
                    self.BATCH.logger.info(out_str)




            ## =============================================================================
            ## BATCHES
            ## =============================================================================


            def BATCH_001_001_ica(self,ASK=True):
                temp_continue = "\nPress any propper key to continue... "

                self.info()
                if ASK & sys.stdout.isatty(): input(temp_continue)

                self.read_raw_data()
                self.check_chans_number()
                self.check_BAD_chans_file()
                self.average_reference_projection()
                self.process_events_and_annots()
                self.bandpass_filter()
                self.plot_channels_power_spectral_density(average=False,exclude=False,)
                self.plot_raw_data_timeseries(total=True,exclude=False,)
                self.extract_metadata_for_events_acquired()
                self.construct_epochs()
                self.check_BAD_epochs_file()

                if sys.stdout.isatty(): plt.close("all")
                self.inspect_epochs()
                if ASK & sys.stdout.isatty(): input(temp_continue)
                if sys.stdout.isatty(): plt.close("all")

                self.export_BAD_epochs_info()
                self.plot_epochs_drop_log()
                self.plot_epochs_AGGREGATED()
                self.plot_epochs_BUNDLES()
                self.construct_evoked()
                self.plot_evoked(evoked0="evoked0")
                self.run_ica()

                if sys.stdout.isatty(): plt.close("all")
                self.inspect_components()
                if ASK & sys.stdout.isatty(): input(temp_continue)
                if sys.stdout.isatty(): plt.close("all")

                self.apply_projections_and_interpolate_bads()

                self.plot_epochs_AGGREGATED(epochs0="epochs2")
                if sys.stdout.isatty(): plt.close("all")

                self.plot_epochs_BUNDLES   (epochs0="epochs2")
                if sys.stdout.isatty(): plt.close("all")

                self.construct_evoked(evoked0="evoked2",epochs0="epochs2")
                self.plot_evoked(evoked0="evoked2",apply_projections=True,interpolate_bads=False,)

                chans_list =  [ "C3" , "C4" , "F3" , "F4" , "PO3" , "PO4" , ]
                DS0.dataBase[IDX].plot_evoked_chans(evoked0="evoked2",chans_list=chans_list,)
                if sys.stdout.isatty(): plt.close("all")

                self.construct_evoked_resp_word_length(evoked0="evoked8",epochs0="epochs2")
                self.plot_evoked(evoked0="evoked8",apply_projections=True,interpolate_bads=False,evoked_name="word_len",)
                if sys.stdout.isatty(): plt.close("all")

                self.export_evoked_as_dataframe(evoked0="evoked2",evoked_name="default",df0_name="dfEvoked2",)
                self.export_epochs_as_dataframe(epochs0="epochs2",events0="events0",df0_name="dfEpochs2",)

                self.export_dataset_as_hickle()


            ## =============================================================================
            ## EEG signal preprocessing functions go below this point
            ## =============================================================================

            ## TODO FIXME CONSIDER adding "if_extn" property and
            ## a case switch for loading variety of file types
            ## using just one function

            def read_raw_data(
                    self,
                    raw0    = "raw0",
                    preload = True,
                    verbose = None,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"reading raw data...")
                if_path = self.locs.if_path
                of_stem = self.locs.of_stem
                of_base = self.locs.of_base
                of_done = self.locs.of_done
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"if_path: {}"   .format(repr(str( if_path ))))
                self.BATCH.logger.info (space0[1]+"of_stem: {}"   .format(repr(str( of_stem ))))
                self.BATCH.logger.info (space0[1]+"of_base: {}"   .format(repr(str( of_base ))))
                self.BATCH.logger.info (space0[1]+"of_done: {}"   .format(repr(str( of_done ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"EXEC: {}"      .format("mne.io.read_raw_brainvision()"))
                ARGS = dict(
                    vhdr_fname = if_path,
                    eog        = ['HEOGL','HEOGR','VEOGb'],
                    misc       = 'auto',
                    scale      = 1.0,
                    preload    = preload,
                    verbose    = verbose,
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )

                ## Re-create dictionary
                ## Start from the scratch
                ## because fresh raw data is being loaded
                self.BATCH.logger.info (space0[1]+"re-creating data dictionary...")
                self.data = OrderedDict()

                self.BATCH.logger.info (space0[1]+"loading data...")
                self.data[raw0] = mne.io.read_raw_brainvision(
                    **ARGS,
                )

                self.BATCH.logger.info (space0[1]+"updating dataset description...")
                self.data[raw0].info["description"] = str(of_stem)
                self.BATCH.logger.info (space0[1]+"DONE...")


            ## TODO TEST ME
            def read_hkl_data(
                    self,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"reading raw data ")
                if_path = self.locs.if_path
                of_stem = self.locs.of_stem
                of_base = self.locs.of_base
                of_done = self.locs.of_done
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"if_path: {}"   .format(repr(str( if_path ))))
                self.BATCH.logger.info (space0[1]+"of_stem: {}"   .format(repr(str( of_stem ))))
                self.BATCH.logger.info (space0[1]+"of_base: {}"   .format(repr(str( of_base ))))
                self.BATCH.logger.info (space0[1]+"of_done: {}"   .format(repr(str( of_done ))))
                self.BATCH.logger.info (space0[1]+"EXE:     {}"   .format("hkl.load()"))
                ARGS = dict(
                    fileobj = if_path,
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.data = hkl.load(
                    **ARGS,
                )


            def check_chans_number(
                    self,
                    raw0 = "raw0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"checking consistency for number of channels...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0 ))))


                self.BATCH.logger.info (space0[1]+"getting expected number of channels from SETUP...")
                chans_EXP = self.BATCH.dataBase.setup["chans"]["init"]
                self.BATCH.logger.debug(space0[2]+"got chans_EXP: " + str(chans_EXP))

                self.BATCH.logger.info (space0[1]+"getting number of channels from DATA...")
                chans_ACT = len(self.data[raw0].copy().pick_types(meg=False,eeg=True).ch_names)
                self.BATCH.logger.debug(space0[2]+"got chans_ACT: " + str(chans_ACT))

                assert chans_EXP == chans_ACT, "PROBLEM: {}".format(
                    " ".join([
                        "Problem occured",
                        "while reading '{}'.".format(str(self.data[raw0])),
                        "It was expected to contain {} EEG channels,".format(chans_EXP,),
                        "but {} were found!".format(chans_ACT,),
                    ])
                )
                self.BATCH.logger.info (space0[1]+"all assertions were met... GREAT!")


            def check_BAD_chans_file(
                    self,
                    raw0 = "raw0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"checking BAD channels information file...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0 ))))
                bad_names    = list()
                of_BAD_chans = self.locs.of_BAD_chans
                self.BATCH.logger.debug(space0[1]+"looking for of_BAD_chans: {}".format(repr(str(of_BAD_chans))))
                if os.path.exists(of_BAD_chans):
                    self.BATCH.logger.info(space0[1]+"found BAD channels file...")
                    with open(of_BAD_chans) as fh:
                        for line in fh:
                            line = line.split('#',1,)[0].strip()
                            if line:
                                bad_names.append(line)

                else:
                    self.BATCH.logger.info(space0[1]+"BAD channels file NOT found...")

                if bad_names:
                    self.BATCH.logger.info (space0[1]+"bad_names: {}".format(str(bad_names)))
                    self.BATCH.logger.info(space0[1]+"adding BAD channels informtion to raw data")
                    self.data[raw0].info['bads'] += bad_names

                    self.BATCH.logger.info(space0[1]+"uniquifying bad channels info")
                    self.data[raw0].info['bads'] = list(set(self.data[raw0].info['bads']))

                else:
                    self.BATCH.logger.warning(space0[1]+"bad_names: {}".format(str(bad_names)))


            def average_reference_projection(
                    self,
                    raw0          = "raw0",
                    montage       = "standard_1005",
                    ref_chans_NEW = "average",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"adding reference projection...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0 ))))

                self.BATCH.logger.info (space0[1]+"adding actual (OLD) reference channel(s) to data...")
                ref_chans_OLD = self.BATCH.dataBase.setup["chans"]["refs"]
                self.BATCH.logger.info (space0[2]+"ref_chans_OLD: {}".format(ref_chans_OLD))
                self.BATCH.logger.info (space0[2]+"EXEC: {}" .format("mne.add_reference_channels()"))
                ARGS = dict(
                    inst         = self.data[raw0],
                    ref_channels = ref_chans_OLD,
                    copy         = False,
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.BATCH.logger.info (space0[2]+"adding {} to DATA".format(repr(str(ref_chans_OLD))))
                mne.add_reference_channels(
                    **ARGS,
                )

                self.BATCH.logger.info (space0[1]+"setting montage...")
                self.BATCH.logger.info (space0[2]+"montage: {}".format(repr(str(montage))))
                self.BATCH.logger.info (space0[2]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"set_montage"))
                ARGS = dict(
                    montage = montage,
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.data[raw0].set_montage(
                    **ARGS,
                )

                self.BATCH.logger.info (space0[1]+"setting average reference projection...")
                self.BATCH.logger.info (space0[2]+"ref_chans_NEW: {}".format(ref_chans_NEW))
                self.BATCH.logger.info (space0[2]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"set_eeg_reference"))
                ARGS = dict(
                    ref_channels = ref_chans_NEW,
                    projection   = True,
                    ch_type      = "eeg",
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.data[raw0].set_eeg_reference(
                    **ARGS,
                )
                self.BATCH.logger.info(space0[1]+"Everything seems to be WELL...")


            def process_events_and_annots(
                    self,
                    raw0    = "raw0",
                    annots0 = "annots0",
                    events0 = "events0",
                    orig0   = "orig0",
                    desc0   = "desc0",
                    time0   = "time0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"initial processing of events and annotations data...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"annots0: {}"   .format(repr(str( annots0 ))))
                self.BATCH.logger.info (space0[1]+"events0: {}"   .format(repr(str( events0 ))))

                self.BATCH.logger.info (space0[1]+"getting annotations from RAW data...")
                self.data[annots0]        = OrderedDict()
                self.data[annots0][orig0] = self.data[raw0].annotations.copy()

                self.BATCH.logger.info (space0[1]+"saving annotations to CSV file...")

                of_suff  = ".{}".format(
                    ".".join([
                        str(whoami()),
                        raw0,
                        annots0,
                        orig0,
                        "csv",
                    ])
                )
                of_name = self.locs.of_base.with_suffix(of_suff)
                self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                self.data[annots0][orig0].save(
                    str(of_name),
                )

                self.BATCH.logger.info (space0[1]+"getting event times and descr. from annotations...")
                self.BATCH.logger.info (space0[2]+"EXEC: {}" .format("mne.events_from_annotations()"))
                ARGS = dict(
                     raw = self.data[raw0],
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                ( temp_event_time,
                  temp_event_desc, ) = mne.events_from_annotations(
                      **ARGS,
                  )
                self.data[events0]        = OrderedDict()
                self.data[events0][desc0] = temp_event_desc
                self.data[events0][time0] = temp_event_time
                self.BATCH.logger.info(space0[1]+"GOOD...")


            def check_for_BAD_spans(
                    self,
                    raw0    = "raw0",
                    annots1 = "annots1",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"checking for BAD spans...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"annots1: {}"   .format(repr(str( annots1 ))))

                of_BAD_spans = self.locs.of_BAD_spans
                self.BATCH.logger.debug(space0[1]+"looking for of_BAD_spans: {}".format(repr(str(of_BAD_spans))))

                # of_annot1 = str(self.locs.of_base.with_suffix(".raw0.annots1.bad_spans.csv"))
                # self.BATCH.logger.info (space0[1]+"looking for of_annot1: "    + str(of_annot1))

                if os.path.exists(of_BAD_spans):
                    self.BATCH.logger.info (space0[2]+"found BAD span annottions file!")
                    self.BATCH.logger.info (space0[2]+"getting BAD span annots data")
                    self.BATCH.logger.info (space0[2]+"EXEC: {}" .format("mne.read_annotations()"))
                    self.data[annots1]              = OrderedDict()
                    self.data[annots1]["bad_spans"] = mne.read_annotations(
                        of_BAD_spans,
                    )
                    self.BATCH.logger.info (space0[2]+"adding BAD span annots to {} data".format(repr(str(raw0))))
                    self.BATCH.logger.info (space0[2]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"set_annotations(OLD+NEW)"))
                    self.data[raw0].set_annotations(
                        self.data[raw0].annotations + self.data[annots1]["bad_spans"],
                    )
                    self.BATCH.logger.info (space0[1]+"ALL GOOD...")
                else:
                    self.BATCH.logger.info (space0[1]+"file not found")
                    self.BATCH.logger.info (space0[1]+"annots1 were NOT updated")
                    self.BATCH.logger.info (space0[1]+"it is OK during the first run")


            def bandpass_filter(
                    self,
                    raw0 = "raw0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"applying bandpass filter to data...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))

                self.BATCH.logger.info (space0[1]+"filter parameters will be taken from the SETUP...")
                self.BATCH.logger.info (space0[1]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"filter"))
                ARGS = dict(
                    l_freq     = self.BATCH.dataBase.setup["filt"]["l_freq"],
                    h_freq     = self.BATCH.dataBase.setup["filt"]["h_freq"],
                    fir_design = self.BATCH.dataBase.setup["filt"]["fir_design"],
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                time_t0 = time.time()
                self.data[raw0].filter(
                    **ARGS,
                )
                time_t1 = time.time()
                time_d1 = time_t1-time_t0
                self.BATCH.logger.info(space0[1]+"time elapsed: " + hf.format_timespan( time_d1 ))
                self.BATCH.logger.info(space0[1]+"pressure is for TYERS...")


            def plot_channels_power_spectral_density(
                    self,
                    raw0    = "raw0",
                    average = False,
                    exclude = True,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting channels power spectral density...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"average: {}"   .format(repr(str( average ))))
                self.BATCH.logger.info (space0[1]+"exclude: {}"   .format(repr(str( exclude ))))

                of_suff = ""
                of_suff = ".".join([of_suff,str(whoami()),raw0])
                of_suff = ".".join([of_suff,"chansAvg" if average else "chansSep"])
                of_suff = ".".join([of_suff,"exclBAD"  if exclude else "inclBAD" ])
                of_suff = ".".join([of_suff,"png"])

                EXCLUDE = self.data[raw0].info["bads"] if exclude else []
                picks   = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=EXCLUDE,)

                self.BATCH.logger.info (space0[1]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"plot_psd"))
                ARGS = dict(
                    show    = False,
                    fmin    =  0,
                    fmax    = 60,
                    picks   = picks,
                    average = average,
                    proj    = True,
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                fig = self.data[raw0].plot_psd(
                    **ARGS,
                )
                fig.set_size_inches(16,8)
                title_old = fig.axes[0].get_title()
                title_new = "{} {} {}\n{} {}".format(
                    title_old,
                    self.locs.of_stem,
                    raw0,
                    "chansAvg" if average else "chansSep",
                    "exclBAD"  if exclude else "inclBAD",
                )
                fig.axes[0].set(title=title_new)
                # plt.tight_layout(pad=.5)
                plt.show()
                of_name = self.locs.of_base.with_suffix(of_suff)
                self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                fig.savefig(of_name, dpi=fig.dpi,)


            def plot_raw_data_timeseries(
                    self,
                    raw0    = "raw0",
                    total   = False,
                    exclude = True,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting raw data timeseries...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"total: {}"     .format(repr(str( total   ))))
                self.BATCH.logger.info (space0[1]+"exclude: {}"   .format(repr(str( exclude ))))

                of_suff = ""
                of_suff = ".".join([of_suff,str(whoami()),raw0])
                of_suff = ".".join([of_suff,"fullSig" if total   else "someSig"])
                of_suff = ".".join([of_suff,"exclBAD" if exclude else "inclBAD"])
                of_suff = ".".join([of_suff,"png"])

                EXCLUDE = self.data[raw0].info["bads"]  if exclude else []
                picks   = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=EXCLUDE,)

                duration = self.data[raw0].times[-1] if total else 20

                fig = self.data[raw0].plot(
                    show       = False,
                    duration   = duration,
                    butterfly  = False,
                    n_channels = len(picks)+2,
                    order      = picks,
                    proj       = True,
                    bad_color = "#ff99cc",
                    )
                fig.set_size_inches(16,8)
                title_old = fig.axes[0].get_title()
                title_new = "{} {} {}\n{} {}".format(
                    title_old,
                    self.locs.of_stem,
                    raw0,
                    "fullSig" if total   else "someSig",
                    "exclBAD" if exclude else "inclBAD",

                )
                ## TODO FIXME no need to update title
                # fig.axes[0].set(title=title_new)

                # plt.tight_layout(pad=.5)
                plt.show()
                if total:
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str(of_name))))
                    fig.savefig(of_name, dpi=fig.dpi,)


            def export_BAD_spans_info(self,raw0="raw0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting BAD spans annotation data to a CSV file...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))

                self.BATCH.logger.info (space0[1]+"extracting BAD spans from RAW data annotations...")
                self.BATCH.logger.info (space0[1]+"for synchronization putpose 0th annot is also included...")
                bads_annot1 = [0]+[ii for ii,an in enumerate(self.data[raw0].annotations) if an['description'].lower().startswith("bad")]

                of_suff0 = ""
                of_suff0 = ".".join([of_suff0,str(whoami()),raw0])
                of_suff1 = ".".join([of_suff0,"annots1","BAD_spans","csv"])
                of_suff2 = ".".join([of_suff0,"annots2","NEW_check","csv"])
                of_name0 = str(self.locs.of_BAD_spans)
                of_name1 = str(self.locs.of_base.with_suffix(of_suff1))
                of_name2 = str(self.locs.of_base.with_suffix(of_suff2))

                self.BATCH.logger.info (space0[1]+"of_name0: {}".format(repr(str(of_name0))))
                self.BATCH.logger.info (space0[1]+"of_name1: {}".format(repr(str(of_name1))))
                self.BATCH.logger.info (space0[1]+"of_name2: {}".format(repr(str(of_name2))))

                self.data[raw0].annotations[bads_annot1].save(of_name0)
                self.data[raw0].annotations[bads_annot1].save(of_name1)
                self.data[raw0].annotations[          :].save(of_name2)


            def extract_metadata_for_events_acquired(
                    self,
                    events0 = "events0",
                    meta0   = "meta0",
                    meta1   = "meta1",
                    time0   = "time0",
                    time1   = "time1",
                    desc0   = "desc0",
                    desc1   = "desc1",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"extracting metadata for events acquired...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"events0: {}"   .format(repr(str( events0 ))))
                self.BATCH.logger.info (space0[1]+"meta0: {}"     .format(repr(str( meta0   ))))
                self.BATCH.logger.info (space0[1]+"meta1: {}"     .format(repr(str( meta1   ))))
                self.BATCH.logger.info (space0[1]+"time0: {}"     .format(repr(str( time0   ))))
                self.BATCH.logger.info (space0[1]+"time1: {}"     .format(repr(str( time1   ))))
                self.BATCH.logger.info (space0[1]+"desc0: {}"     .format(repr(str( desc0   ))))
                self.BATCH.logger.info (space0[1]+"desc1: {}"     .format(repr(str( desc1   ))))

                self.BATCH.logger.info (space0[1]+"extracting metadata for events acquired...")
                self.BATCH.logger.info (space0[2]+"these will be kept in pandas DataFrame...")
                self.BATCH.logger.info (space0[2]+"under meta0: {} key ".format(repr(str( meta0 ))))

                self.data[events0][meta0] = pd.DataFrame(
                    self.data[events0][time0],
                    columns = ["ONSET","DURATION","CODE"],
                )
                self.BATCH.logger.info (space0[2]+"computing onset diff column for {}...".format(repr(str(meta0))))
                # self.data[events0][meta0].loc[:,"DIFF"] = self.data[events0][meta0]["ONSET"].diff()
                self.data[events0][meta0]["DIFF"] = self.data[events0][meta0]["ONSET"].diff()
                self.data[events0][meta0]["DIFF"].fillna(0,inplace=True)

                self.BATCH.logger.info (space0[1]+"selecting exclusively stimuli-related events...")
                self.BATCH.logger.info (space0[2]+"these will be kept in pandas DataFrame...")
                self.BATCH.logger.info (space0[2]+"under meta1: {} key ".format(repr(str( meta1 ))))
                self.data[events0][meta1] = self.data[events0][meta0].copy(deep=True)[
                    (self.data[events0][meta0]["CODE"] > 100) &
                    (self.data[events0][meta0]["CODE"] < 300)
                ]
                self.BATCH.logger.info (space0[2]+"computing onset diff column for {}...".format(repr(str(meta1))))
                self.BATCH.logger.info (space0[2]+"... this (re)computation is requided after...")
                self.BATCH.logger.info (space0[2]+"... removing events not related to stimuli were dropped...")
                # self.data[events0][meta1].loc[:,"DIFF"] = self.data[events0][meta1]["ONSET"].diff()
                self.data[events0][meta1]["DIFF"] = self.data[events0][meta1]["ONSET"].diff()
                self.data[events0][meta1]["DIFF"].fillna(0,inplace=True)

                self.BATCH.logger.info (space0[1]+"merging event details in {} with extra info from SETUP...".format(repr(str( meta1 ))))
                self.BATCH.logger.info (space0[2]+"these will be kept in pandas DataFrame...")
                self.BATCH.logger.info (space0[2]+"under meta1: {} key ".format(repr(str( meta1 ))))
                self.data[events0][meta1] = pd.merge(
                    left      = self.data[events0][meta1],
                    right     = self.BATCH.dataBase.setup["events"]["dgn0"],
                    how       = "left",
                    left_on   = "CODE",
                    right_on  = "CODE",
                    sort      = False,
                    suffixes  = ("_acq","_dgn"),
                    copy      = True,
                    indicator = False,
                    validate  = "m:1",
                )
                self.BATCH.logger.info (space0[2]+"adding extra columns (FILE,SUB,RUN,TASK) to {}...".format(repr(str( meta1 ))))
                self.data[events0][meta1]["FILE"] = str(self.locs.of_stem)
                self.data[events0][meta1]["SUB"]  = get_bids_prop(if_name=str(self.locs.of_stem),prop="sub",)
                self.data[events0][meta1]["RUN"]  = get_bids_prop(if_name=str(self.locs.of_stem),prop="run",)
                self.data[events0][meta1]["TASK"] = get_bids_prop(if_name=str(self.locs.of_stem),prop="task",)


                self.BATCH.logger.info (space0[1]+"exporting event onset information from {}...".format(repr(str(meta1))))
                self.BATCH.logger.info (space0[2]+"these will be kept under time1: {} key..."   .format(repr(str(time1))))

                self.data[events0][time1] = self.data[events0][meta1][["ONSET","DURATION","CODE",]].to_numpy()


                self.BATCH.logger.info (space0[1]+"producing exclusively stimuli-related events descriptions...")
                self.BATCH.logger.info (space0[2]+"these will be kept under desc1: {} key...".format(repr(str(desc1))))
                self.data[events0][desc1] = OrderedDict()
                for (key0, val0) in self.data[events0][desc0].items():
                    if (val0 > 100) & (val0 < 300):
                        self.data[events0][desc1][key0] = val0

                self.BATCH.logger.info (space0[1]+"checking assertions...")
                assert len(self.data[events0][meta1]) == 80       , "PROBLEM: Please fix meta1"
                assert len(self.data[events0][desc1]) == 80       , "PROBLEM: Please fix desc1"
                assert self.data[events0][time1].shape == (80, 3) , "PROBLEM: Please fix time1"
                self.BATCH.logger.info (space0[1]+"all assertions were met... GREAT!")


            def construct_epochs(
                    self,
                    raw0    = "raw0",
                    events0 = "events0",
                    epochs0 = "epochs0",
                    meta1   = "meta1",
                    time1   = "time1",
                    desc1   = "desc1",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"constructing epochs...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"events0: {}"   .format(repr(str( events0 ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))
                self.BATCH.logger.info (space0[1]+"meta1: {}"     .format(repr(str( meta1   ))))
                self.BATCH.logger.info (space0[1]+"time1: {}"     .format(repr(str( time1   ))))
                self.BATCH.logger.info (space0[1]+"desc1: {}"     .format(repr(str( desc1   ))))

                exclude = self.data[raw0].info["bads"] + self.BATCH.dataBase.setup["chans"]["refs"]
                exclude = self.BATCH.dataBase.setup["chans"]["refs"]
                exclude = self.data[raw0].info["bads"]
                exclude = []

                self.BATCH.logger.info (space0[1]+"exclude: {}"   .format(repr(str( exclude ))))

                picks = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=exclude,)

                self.BATCH.logger.info (space0[1]+"getting event times and descr. from annotations...")
                self.BATCH.logger.info (space0[2]+"EXEC: {}" .format("mne.Epochs(...)"))
                ARGS = dict(
                    raw      = self.data[raw0].copy(),
                    events   = self.data[events0][time1],
                    event_id = self.data[events0][desc1],
                    metadata = self.data[events0][meta1],
                    tmin     = -0.200,
                    tmax     =  0.900,
                    baseline = (None, 0),
                    picks    = picks,
                    preload  = False, ## TODO FIXME CHECKUP THAT
                    reject_by_annotation=True,
                    reject   = self.BATCH.dataBase.setup["params"]["reject"],
                    flat     = self.BATCH.dataBase.setup["params"]["flat"],
                    decim    = 5,
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.data[epochs0] = mne.Epochs(
                      **ARGS,
                )
                self.BATCH.logger.info (space0[1]+"DONE with epochs...")


            def check_BAD_epochs_file(
                    self,
                    epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"checking BAD epochs information file...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                bad_epochs = list()
                of_BAD_epochs   = self.locs.of_BAD_epochs
                self.BATCH.logger.debug(space0[2]+"looking for of_BAD_epochs: {}".format(str(repr(of_BAD_epochs))))
                if os.path.exists(of_BAD_epochs):
                    self.BATCH.logger.info(space0[2]+"found bad epochs file...")
                    with open(of_BAD_epochs) as fh:
                        for line in fh:
                            line = line.split('#',1,)[0].strip()
                            if line:
                                bad_epochs.append(int(line))

                bad_epochs = list(set(bad_epochs))
                self.BATCH.logger.info (space0[2]+"bad_epochs: {}".format(repr(str(bad_epochs))))
                if bad_epochs:
                    self.BATCH.logger.info(space0[1]+"adding BAD epochs informtion to data")
                    self.data[epochs0].drop(bad_epochs)
                    ## TODO FIXME check if this truely operates inplace (in-place)


            def inspect_epochs(
                    self,
                    epochs0 = "epochs0",
                    exclude = True,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"inspecting epochs (possibly MARKING BAD)")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))
                self.BATCH.logger.info (space0[1]+"exclude: {}"   .format(repr(str( exclude ))))

                EXCLUDE = self.data[epochs0].info["bads"] if exclude else []
                picks   = mne.pick_types(self.data[epochs0].info,meg=False,eeg=True,exclude=EXCLUDE,)

                self.BATCH.logger.info (space0[2]+"plotting epochs")
                self.data[epochs0].plot(
                    n_channels = len(picks)+2,
                    n_epochs=4,
                )
                self.BATCH.logger.info (space0[1]+"DONE")


            def drop_BAD_epochs(
                    self,
                    epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"dropping epochs marked as BAD")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                self.data[epochs0].drop_bad()
                ## TODO FIXME check if this truely operates inplace (in-place)

                self.BATCH.logger.info (space0[1]+"DONE")


            def export_BAD_epochs_info(
                    self,
                    epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting BAD epochs annotation data to a CSV file")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                drop_log = dc(self.data[epochs0].drop_log)
                drop_idx = [this for this in [idx for idx,item in enumerate(drop_log) if item] if drop_log[this][0]=="USER"]
                of_BAD_epochs = self.locs.of_BAD_epochs
                self.BATCH.logger.info (space0[1]+"of_BAD_epochs: {}".format(repr(str(of_BAD_epochs))))
                self.BATCH.logger.info (space0[1]+"drop_idx: {}".format(repr(str(drop_idx))))
                if drop_idx:
                    with open(of_BAD_epochs, 'w') as fh:
                        for item in drop_idx:
                            fh.write("{} # USER\n".format(item))
                self.BATCH.logger.info (space0[1]+"DONE")


            def plot_epochs_drop_log(
                    self,
                    epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting epoch drop log hist")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                fig = self.data[epochs0].plot_drop_log(show=False)
                fig.set_size_inches(8,4)

                title_old = fig.axes[0].get_title()
                title_new = "{}\n{} {}".format(
                    self.locs.of_stem,
                    epochs0,
                    title_old,
                )
                fig.axes[0].set(title=title_new)
                plt.show()
                of_suff = ""
                of_suff = ".".join([of_suff,str(whoami()),epochs0])
                of_suff = ".".join([of_suff,"0"])
                of_suff = ".".join([of_suff,"png"])
                of_name = self.locs.of_base.with_suffix(of_suff)
                self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                fig.savefig(of_name, dpi=fig.dpi,)


            def plot_epochs_AGGREGATED(self,epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting epochs AGGREGATED")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))

                combines = ["gfp","mean",]
                for jj,combine in enumerate(combines):
                    # .copy().apply_proj()
                    ## TODO FIXME consider the above projection
                    figs = self.data[epochs0].plot_image(
                        show     = False,
                        group_by = None,
                        combine  = combine,
                        sigma    = 0,
                    )
                    for ii,fig in enumerate(figs):
                        fig.set_size_inches(16,8)
                        title_orig = fig.axes[0].get_title()
                        title_pref = str(self.locs.of_stem)
                        fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
                        plt.show()
                        of_suff  = ""
                        of_suff += "."+epochs0
                        of_suff += ".plot_image.aggAllChans.0-{}-{}-{}.png".format(jj,combine,ii,)
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                        fig.savefig(of_name, dpi=fig.dpi,)


            def plot_epochs_BUNDLES(self,epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting epochs BUNDLES")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))

                elec_idx = OrderedDict()
                for key,val in self.BATCH.dataBase.setup["chans"]["bund0"].items():
                    elec_idx[key] = mne.pick_types(
                        self.data[epochs0].info,
                        meg       = False,
                        eeg       = True,
                        exclude   = [],
                        selection = val,
                    )

                combines = ["gfp","mean",]
                for jj,combine in enumerate(combines):
                    figs = self.data[epochs0].plot_image(
                        show     = False,
                        group_by = elec_idx,
                        combine  = combine,
                        sigma    = 0,
                    )
                    for ii,fig in enumerate(figs):
                        fig.set_size_inches(16,8)
                        title_orig = fig.axes[0].get_title()
                        title_pref = str(self.locs.of_stem)
                        fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
                        title_bndl = title_orig.partition(" ")[0]
                        plt.show()
                        of_suff  = ""
                        of_suff += "."+epochs0
                        of_suff += ".plot_image.aggChanBundles.0-{}-{}-{}-{}.png".format(jj,combine,ii,title_bndl,)
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                        fig.savefig(of_name, dpi=fig.dpi,)


            def construct_evoked(self,evoked0="evoked0",epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"constructing evoked")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"evoked0: "    + str(evoked0))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.data[evoked0]            = OrderedDict()
                self.data[evoked0]["default"] = OrderedDict()
                self.data[evoked0]["default"]["all_in"] = self.data[epochs0].average()
                for ii,(key,val) in enumerate(self.BATCH.dataBase.setup["queries"]["default"].items()):
                    self.data[evoked0]["default"][key]   = self.data[epochs0][val].average()


            def construct_evoked_resp_word_length(self,evoked0="evoked0",epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"constructing evoked with respect to word length")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"evoked0: "    + str(evoked0))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.data[evoked0]             = OrderedDict()
                self.data[evoked0]["word_len"] = OrderedDict()
                query = 'LEN == {}'

                for n_letters in sorted( self.data[epochs0].metadata["LEN"].unique()):
                    self.data[evoked0]["word_len"][str(n_letters)] = self.data[epochs0][query.format(n_letters)].average()


            def plot_evoked(self,evoked0="evoked0",apply_projections=True,interpolate_bads=True,evoked_name="default"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting evoked")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"evoked0: "    + str(evoked0))

                spatial_colors = False
                spatial_colors = True
                ## Time points for which topomaps will be displayed
                times  = "auto"
                times  = list()
                times += [-0.200,-0.100,0,]
                times += self.BATCH.dataBase.setup["time"]["means0"].values()
                # times += list(sum(self.BATCH.dataBase.setup["time"]["spans0"].values(), ()))
                times = sorted(times)
                for ii,(key,evoked) in enumerate(self.data[evoked0][evoked_name].items()):
                    title = ""
                    title += "EEG ({}) ".format( evoked.info["nchan"] - len(evoked.info["bads"]) )
                    title += str(self.locs.of_stem)
                    evoked = evoked.copy()
                    if apply_projections:
                        evoked = evoked.apply_proj()

                    if interpolate_bads:
                        evoked = evoked.interpolate_bads(
                        reset_bads=True,
                        mode="accurate",
                        )

                    fig = evoked.plot_joint(
                        title   = title + "\n{} ({})".format(key,evoked0),
                        times   = times,
                        ts_args = dict(
                            time_unit      = "s",
                            ylim           = dict(eeg=[-15, 15]),
                            spatial_colors = spatial_colors,
                        ),
                        topomap_args = dict(
                            cmap     = "Spectral_r",
                            outlines = "skirt",
                        ),
                    )
                    fig.set_size_inches(16,8)
                    plt.show()
                    of_suff  = ""
                    of_suff += "."+evoked0
                    of_suff += ".plot_joint.0-{}-{}.png".format(ii,key,)
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    fig.savefig(of_name, dpi=fig.dpi,)


            def plot_evoked_chans(self,evoked0="evoked0",chans_list=[],evoked_name="default",):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting evoked for some data channels accross conditions")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"evoked0: "    + str(evoked0))
                self.BATCH.logger.info (space0[1]+"chans_list: " + str(chans_list))

                for ii,item in enumerate(chans_list):
                    title =  str(self.locs.of_stem) + ": " + item + " ({})".format(evoked0)
                    ## Normally a list of figures is created
                    ## Here (for each iteration) we produce list with exactly one item
                    figs = mne.viz.plot_compare_evokeds(
                        evokeds  = self.data[evoked0][evoked_name],
                        picks    = item,
                        ci       = True,
                        ylim     = dict(eeg=[-12,12]),
                        invert_y = True,
                        title    = title,
                    )
                    figs[0].set_size_inches(16,8)
                    plt.show()
                    of_suff  = ""
                    of_suff += "."+evoked0
                    of_suff += ".plot_compare_evokeds.0-{}-{}.png".format(ii,item,)
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    figs[0].savefig(of_name, dpi=figs[0].dpi,)


            def run_ica(self,ica0="ica0",epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"running ICA")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.BATCH.logger.info (space0[1]+"ica0: "    + str(ica0))

                random_states = list()
                of_rand       = self.locs.of_rand
                self.BATCH.logger.debug(space0[1]+"looking for of_rand: " + str(of_rand))
                if os.path.exists(of_rand):
                    self.BATCH.logger.info(space0[1]+"found random state file...")
                    with open(of_rand) as fh:
                        for line in fh:
                            line = line.split('#',1,)[0].strip()
                            if line:
                                random_states.append(int(line))

                else:
                    self.BATCH.logger.info(space0[1]+"BAD channels file NOT found...")
                    random_states = [0]


                time_T0 = time.time()
                self.data[ica0] = mne.preprocessing.ica.ICA(
                    n_components       = 50,
                    n_pca_components   = 50,
                    max_pca_components = 50,
                    method             = 'fastica',
                    max_iter           = 1600,
                    # noise_cov          = noise_cov,
                    random_state       = random_states[0],
                ).fit(
                    self.data[epochs0],
                )
                time_T1 = time.time()
                time_D1 = time_T1-time_T0
                self.BATCH.logger.info (space0[1]+"time elapsed: " + hf.format_timespan( time_D1 ))


            def inspect_components(self,ica0="ica0",epochs0="epochs0"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"inspecting ICs")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.BATCH.logger.info (space0[1]+"ica0: "       + str(ica0))

                title = "ICA Componentts\n"
                title += str(self.locs.of_stem)
                # TODO FIXME This can also output a single figure object
                # (not necesarily a list of figs)
                figs = self.data[ica0].plot_components(
                    inst   = self.data[epochs0],
                    title  = title,
                )
                for ii,fig in enumerate(figs):
                    fig.set_size_inches(16,16)
                    plt.show()
                    of_suff  = ""
                    of_suff += "."+ica0
                    of_suff += ".plot_components.win.0-{}.png".format(ii)
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    fig.savefig(of_name, dpi=fig.dpi,)


            def plot_components(self,ica0="ica0",epochs0="epochs0",rejected=True,save=False):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting ICs")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.BATCH.logger.info (space0[1]+"ica0: "       + str(ica0))
                self.BATCH.logger.info (space0[1]+"rejected: "   + str(rejected))
                self.BATCH.logger.info (space0[1]+"save: "       + str(save))

                exclude = sorted(self.data[ica0].exclude)
                include = [item for item in list(range(self.data[ica0].n_components)) if not item in exclude]
                picks = exclude if rejected else include
                if picks:
                    figs = self.data[ica0].plot_properties(
                        inst  = self.data[epochs0],
                        picks = picks,
                    )
                    for ii,fig in enumerate(figs):
                        fig.set_size_inches(16,16)
                        title_orig = fig.axes[0].get_title()
                        title_pref = str(self.locs.of_stem)
                        fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
                        plt.show()
                        if save:
                            of_suff  = ""
                            of_suff += "."+ica0
                            of_suff += ".plot_properties"
                            of_suff  = ".".join([of_suff,"exc" if rejected else "inc"])
                            of_suff += ".0-{:02d}.png".format(ii)
                            of_name = self.locs.of_base.with_suffix(of_suff)
                            self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                            fig.savefig(of_name, dpi=fig.dpi,)



            def apply_projections_and_interpolate_bads(self,ica0="ica0",epochs0="epochs0",epochs1="epochs1",epochs2="epochs2"):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"applying ICA and reference projections, and interpolating bads")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"ica0: "       + str(ica0))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                self.BATCH.logger.info (space0[1]+"epochs1: "    + str(epochs1))
                self.BATCH.logger.info (space0[1]+"epochs2: "    + str(epochs2))

                self.data[epochs0].load_data()

                self.data[epochs1] = self.data[ica0].apply(
                    self.data[epochs0].copy(),
                )
                reset_bads = True
                mode       = "accurate"

                self.BATCH.logger.info (space0[1]+"reset_bads: "    + str(reset_bads))
                self.BATCH.logger.info (space0[1]+"mode: "          + str(mode))

                self.data[epochs2] = self.data[epochs1].copy(
                ).apply_proj(
                ## ).resample(
                ##     sfreq=200,
                ).interpolate_bads(
                    reset_bads = reset_bads,
                    mode       = mode,
                )
                self.BATCH.logger.debug("="*77)
                self.BATCH.logger.debug(self.data[epochs0].info)
                self.BATCH.logger.debug("="*77)
                self.BATCH.logger.debug(self.data[epochs1].info)
                self.BATCH.logger.debug("="*77)
                self.BATCH.logger.debug(self.data[epochs2].info)


            def export_dataset_as_hickle(self,):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting dataset as hickle")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                # self.of_done = self.of_base.with_suffix(".gzip.hkl")
                # self.of_done
                # self.of_base.with_suffix(".gzip.hkl")
                hkl.dump(
                    self,
                    self.locs.of_done,
                    mode="w",
                    compression="gzip",
                )
                # DATA[777] = hkl.load(DATA[IDX].src[0].of_base+".gzip.hkl")
                # DATA[777] = hkl.load("data/s0000_raw/sub-42aszm/eeg/sub-42aszm_task-lexdec_run-001.gzip.hkl")
                # IDX = 777


            def export_evoked_as_dataframe(self,evoked0="evoked0",evoked_name="default",df0_name=None,):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting evoked as pandas dataframe")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"evoked0: "    + str(evoked0))
                self.BATCH.logger.info (space0[1]+"evoked_name: " + str(evoked_name))
                ## TODO FIXME add assertion for datatype (EVOKED)
                df0 = pd.DataFrame()

                for ii,(key,evoked) in enumerate(self.data[evoked0][evoked_name].items()):
                    temp0 =  evoked.to_data_frame(
                        time_format=None,
                        long_format=True,
                    )
                    temp0["STEM"] = str(self.locs.of_stem)
                    temp0["KEY"]  = key
                    self.BATCH.logger.info (space0[1]+"PROC: " + str(key))
                    df0 = df0.append(temp0)


                self.BATCH.logger.info (space0[1]+"DIMS: " + str(df0.shape))
                with pd.option_context("display.max_colwidth"         ,  200):
                    with pd.option_context("display.width"            , 3500):
                        with pd.option_context("display.max_rows"     ,   45):
                            with pd.option_context("display.min_rows" ,   45):
                                self.BATCH.logger.info (space0[1]+"SAMP:\n" + str(df0.sample(n=12).sort_index()))

                of_suff  = ""
                of_suff += "."+evoked0
                of_suff += "."+evoked_name
                of_suff += ".pandas_evoked.csv"
                of_name = self.locs.of_base.with_suffix(of_suff)

                df0.to_csv(
                    of_name,
                    index = False,
                )

                if df0_name is not None:
                    self.data[df0_name] = df0


            def export_epochs_as_dataframe(self,epochs0="epochs0",events0="events0",df0_name=None,):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting epochs as pandas dataframe")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                self.BATCH.logger.info (space0[1]+"epochs0: "    + str(epochs0))
                ## TODO FIXME add assertion for datatype (EPOCHS)
                df0 = self.data[epochs0].to_data_frame(
                    time_format = None,
                    index       = None,
                    long_format = True,
                )
                df0 = df0.join(
                    df0["condition"].str.split('/',expand=True).rename(columns={0:"TYPE",1:"STIM"})
                )
                df0["STIM"] = df0["STIM"].str[1:].astype(int)

                del df0["TYPE"]

                if True:
                    df0 = pd.merge(
                        left      = df0,
                        right     = self.data[events0][meta1],
                        how       = "left",
                        left_on   = ["STIM"],
                        right_on  = ["CODE"],
                        sort      = False,
                        suffixes  = ("_orig","_meta"),
                        copy      = True,
                        indicator = False,
                        # validate  = "1:m",
                    )
                    # self.BATCH.dataBase.setup["chans"]["bund0"].items()
                    di0 = self.BATCH.dataBase.setup["chans"]["bund0"]
                    di1 = OrderedDict()
                    for key,val in di0.items():
                        for item in val:
                            di1[item] = key

                    df0["BUNDLE"] = df0["channel"].apply(lambda x: di1[x] if x in di1 else None)

                of_suff  = ""
                of_suff += "."+epochs0
                of_suff += ".pandas_epochs.csv"
                of_name = self.locs.of_base.with_suffix(of_suff)

                df0.to_csv(
                    of_name,
                    index = False,
                )

                if df0_name is not None:
                    self.data[df0_name] = df0
