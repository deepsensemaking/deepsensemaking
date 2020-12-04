#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
deepsensemaking (dsm) eeg/mne auxiliary tools



TODO
- [ ] add key="latest0" to dataBase[IDX].data dictionary (DataBase)
      - possibly for "Evoked"
      - that will work alongside with keys previously used
        for intermediate steps
      - this way we can keep saving to the same file
      - use other file as flag for DONE (see below)

- [X] fully implement "doneSuffix"

- [X] add of_stem to ICs

- [X] add data unloading to JOBS

- [ ] do epoch rejection in two separate phases (before and after ICA)
      - reject nothing with "construct_epochs()"
        or be very generous (like 5000 uV)
      - next use "drop_BAD_epochs()"
        - reject = self.BATCH.dataBase.setup["params"]["reject"],
        - flat   = self.BATCH.dataBase.setup["params"]["flat"],

- [ ] add to DataBase a method ".info_loaded()"
      - OR add parameter(s) to ".info()"
        - "loaded"
        - "brief"
        - "list='ica8'" return list containing dataset numbers that have ".data["ica8"] item"

- [ ] add carpet plot with channel-wise smoothing

- [ ] drag along files like *.BAD_chans (starting from raw data directories?)


# Code snippet for testing puropose goes below this point

get_ipython().magic("run st.001.001.ica.tg.ipy")
for IDX in [0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,19]:
    DS0.dataBase[IDX].read_raw_data()
    DS0.dataBase[IDX].check_chans_number()
    DS0.dataBase[IDX].check_BAD_chans_file()
    DS0.dataBase[IDX].average_reference_projection()
    DS0.dataBase[IDX].process_events_and_annots()
    DS0.dataBase[IDX].bandpass_filter()
    DS0.dataBase[IDX].extract_metadata_for_events_acquired()
    DS0.dataBase[IDX].construct_epochs(
        raw0    = "raw0",
        events0 = "events0",
        epochs0 = "epochs0",
        meta1   = "meta1",
        time1   = "time1",
        desc1   = "desc1",
        reject  = None,
        flat    = None,
        preload = False,
    )





"""

import os
import sys
import glob
import pathlib
import shutil

import json
import numpy  as np
import pandas as pd
import hickle as hkl

import warnings
import logging

import inspect

"""
checkup:


"""

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.colors import ListedColormap

import seaborn as sns

from scipy import ndimage
from scipy.signal import find_peaks, peak_prominences


# plt.ion()
plt.ioff()

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

from deepsensemaking.bids  import get_bids_prop
from deepsensemaking.dicts import str_dict,print_dict

import itertools

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
    """MNE batch job class...
    (AKA: duct-tape and cable-ties for EEG preprocessing)
    Example usage:

      DS0 = BatchMNE(
          objName     = "DS0",
          sourceDir   = "../../rawdata",
          targetDir   = "../../derivatives/preproc002/JOB_000",
          globSuffix  = "sub-*/ses-*/eeg/sub-*_task-lexdec_run-*.vhdr",
          doneSuffix  = ".JOB_000.DONE",
          setupFile   = "../../rawdata/misc/setup.json",
          stimuliFile = "../../rawdata/stimuli/task-lexdec_stimuli.MNE.csv",
          verbose     = 0,
      )

      DS0.dataBase.setup.read()

      DS0.inputPaths.glob()
      DS0.inputPaths.select("selector__keep_subjects_004.txt",mode="keep")
      DS0.dataBase.get_paths()
      DS0.info()

      DS0.dataBase.info()

    """

    def __init__(
            self,
            objName     = "DS",
            sourceDir   = "rawdata",
            targetDir   = "derivatives",
            globSuffix  = "sub-*/ses-*/eeg/sub-*.vhdr",
            doneSuffix  = ".DONE",
            setupFile   = "setup.json",
            stimuliFile = "stimuli.csv",
            verbose     = 0,
    ):

        ## Ensure that type of all paths is pathlib.Path
        sourceDir   = pathlib.Path(sourceDir)
        targetDir   = pathlib.Path(targetDir)
        globSuffix  = pathlib.Path(globSuffix)
        # doneSuffix  = pathlib.Path(doneSuffix)
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
        self.doneSuffix  = doneSuffix
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

        ## Also attach MNE logger
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
        out_str += space1[1]+self.objName+".doneSuffix  = "+repr(str(self.doneSuffix ))
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
            ## FUTURE: CONSIDER: using INSP depth for logs formatting
            # self.DPTH = len(self.INSP)-2
            self.BATCH   = BATCH




        def glob(self):
            self.BATCH.logger.info(space0[0]+"RUNNING: {}.{}".format(
                ".".join(self.INSP),
                str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"globbing for input data paths...")
            self.BATCH.logger.info(space0[1]+"pattern: {}".format(repr(str(self.BATCH.globPattern))))
            ## Glob for files that match provided pattern
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
                    # line = line.strip()
                    line = line.split('#',1,)[0].strip()
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
                space0[1]+"{} contains {} items".format(
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
            out_str += space0[1]+"{} contains {} items".format(
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
            self.INSP    = [item for item in INSP]+[objName]
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




        def read_ALL_hkl(self,raw0="raw0",):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"getting data for ALL paths")
            for idx,item in enumerate(self.data):
                self.BATCH.logger.info(space0[1]+"reading HKL data: {}".format(item))
                self.data[idx].read_hkl()



        def write_ALL_hkl(self,mark_DONE=True):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            self.BATCH.logger.info(space0[1]+"writing ALL data to HKL files")
            for idx,item in enumerate(self.data):
                self.BATCH.logger.info(space0[1]+"writing HKL data: {}".format(item))
                self.data[idx].write_hkl()
                if mark_DONE:
                    self.data[idx].mark_DONE()



        def update_indices(self,):
            for idx0,item0 in enumerate(self.data):
                item0.idx       = idx0
                item0.INSP[-1]  = item0.INSP_TEMP[-1]+"[{}]".format(idx0)
                item0.locs.INSP = [item for item in item0.INSP]+[item0.locs.objName]




        def keep_concatenated(self,):
            self.data = [ item0 for item0 in self.data if item0.concatBool ]




        def clean_ALL_concatDict(self,):
            for idx0,item0 in enumerate(self.data):
                for item1 in item0.concatDict.values():
                    item1 = None




        def concatenate_data_by_subject(self,signals0="signals0",):
            """
            Merge datasets
            This is a quick-fix (quick-hack) implemented here
            because in the initial pipeline design each run was
            preprocessed/analysed separately, however

            TODO FIXME add information about RUN in the EVENTS description.
            TODO FIXME CONSIDER better use such merging at the stage of getting paths

            Example use:
              data = DS0.dataBase
              data.concatenate_data_by_subject(signals0="raw0",)
              data.concatenate_data_by_subject(signals0="epochs0",)


            """
            ## Produce data dictionary
            ## - keys are based on subject codes
            ## - values contain lists that in turn contain
            ##   indices of datasets for a given subject
            self.group_by_subject = OrderedDict()
            for idx0,item0 in enumerate(self.data):
                self.BATCH.logger.info (space0[1]+"looking at dataset {:03d}: {}".format(idx0,item0.locs.of_stem))
                self.BATCH.logger.debug(space0[2]+"sub        : {}".format(item0.sub))
                self.BATCH.logger.debug(space0[2]+"run        : {}".format(item0.run))
                self.BATCH.logger.debug(space0[2]+"concatBool : {}".format(item0.concatBool))
                ## check if item is an original recording
                ## not a concatenation
                if not item0.concatBool:
                    ## check if subject for this item is already used as a key
                    if not item0.sub in list(self.group_by_subject.keys()):
                        self.group_by_subject[item0.sub] = list()

                    ## append index for this item to list sored in dict value
                    self.group_by_subject[item0.sub].append(idx0)

            ## Construct new datasets
            ## - iterating over subjects
            for sub0,list0 in self.group_by_subject.items():
                ## - get the first item
                ## - use it as a templete for the contcatenated dataset
                item0 = self.data[list0[0]]
                ## - make run-000
                ##   - TODO FIXME add regex here to facilitate
                if_path = pathlib.Path(str(item0.locs.if_path).replace("_run-001","_run-000",))
                self.data.append(
                    self.DataSet(
                        BATCH      = self.BATCH,
                        INSP       = self.INSP ,
                        item       = if_path,
                        idx        = len(self.data),
                        concatBool = True,
                        concatDict = OrderedDict(),
                    )
                )
                self.BATCH.logger.info (space0[1]+"ADDED DataSet: {}".format(repr(str(self.data[-1]))))
                self.BATCH.logger.info (space0[2]+"BASED on: {}"     .format(repr(str(item0))))
                ## - iterating over signals for current subject
                for idx1 in list0[:]:
                    item1 = self.data[idx1]
                    if signals0 in item1.data.keys():
                        self.BATCH.logger.info (space0[3]+"INCLUDING {} from DataSet: {}".format(repr(str(signals0)),item1))
                        self.data[-1].concatDict[str(item1.locs.of_stem)] = dc(item1.data[signals0])

                concatList = list(self.data[-1].concatDict.values())
                if concatList:
                    if isinstance(concatList[0], mne.epochs.BaseEpochs):
                        self.data[-1].data[signals0] = mne.concatenate_epochs(dc(concatList))

                    elif isinstance(concatList[0], mne.io.brainvision.brainvision.RawBrainVision):
                        self.data[-1].data[signals0] = mne.concatenate_raws(dc(concatList))

                    else:
                        raise NotImplementedError




        def JOB_000(self,ASK=False,cleanup=True):
            self.BATCH.logger.info(
                space0[0]+"RUNNING: {}.{}".format(
                    ".".join(self.INSP),
                    str(whoami()),
            ))
            plt.ioff()
            for idx,item in enumerate(self.data):
                if not item.locs.of_data.is_file():
                    self.BATCH.logger.info(space0[1]+"PROCESSING: [{}] {}".format(idx,item,))
                    self.data[idx].JOB_000(ASK=ASK,cleanup=cleanup)




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
                        repr(str(key0)),
                        repr(str(key1)),
                )
                out_str += "\n\n" # TODO FIXME add four spaces to the left of each line
                with pd.option_context("display.max_colwidth"         ,  200):
                    with pd.option_context("display.width"            , 3500):
                        with pd.option_context("display.max_rows"     ,   45):
                            with pd.option_context("display.min_rows" ,   45):
                                out_str += pf(self[key0][key1])

                self.BATCH.logger.info(out_str)




        class DataSet():




            def __init__(self,BATCH,INSP,item,idx,concatBool=False,concatDict=None):
                self.BATCH      = BATCH
                self.INSP_TEMP  = [item for item in INSP]
                self.INSP       = [item for item in INSP]
                self.INSP[-1]   = self.INSP[-1]+"[{}]".format(idx)
                self.idx        = idx
                self.locs       = self.Locs(self.BATCH,self.INSP,item,objName="locs")
                self.sub        = get_bids_prop(self.locs.of_stem,"sub",)
                self.ses        = get_bids_prop(self.locs.of_stem,"ses",)
                self.task       = get_bids_prop(self.locs.of_stem,"task",)
                self.run        = get_bids_prop(self.locs.of_stem,"run",)
                self.concatBool = concatBool
                self.concatDict = concatDict
                self.data       = OrderedDict()
                os.makedirs(self.locs.od_path,mode=0o700,exist_ok=True,)




            def __repr__(self):
                return self.__str__()




            def __str__(self):
                return str(self.locs.of_stem) + " ("+str(len(self.get_keys()))+")"




            def __len__(self):
                return len(self.data.keys())




            def get_keys(self):
                return list(self.data.keys())




            def info(self,level=None):
                out_str  = ""
                out_str += space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                )
                out_str += "\n"
                out_str += space1[1]+self.__str__()
                out_str += space1[1]+str(self.get_keys())
                out_str += "\n"
                if level is not None:
                    out_str += str_dict(
                        self.data,
                        " "*4+".".join(self.INSP)+".data",
                        max_level=level,
                        max_len=77,
                    )

                self.BATCH.logger.info(out_str)




            def mark_DONE(self):
                of_name = self.locs.of_done
                self.BATCH.logger.info(space0[1]+"job DONE for {} ".format(repr(str(self.locs.of_stem))))
                self.BATCH.logger.info(space0[1]+"file: {}".format(repr(str(of_name))))
                with open(of_name, 'w') as fh:
                    pass




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
                    self.of_data       = self.of_base.with_suffix(".gzip.hkl")
                    self.of_done       = self.of_base.with_suffix(BATCH.doneSuffix)


                def __repr__(self):
                    return self.__str__()


                def __str__(self):
                    out_str  = ""
                    out_str += space1[2]+"      if_path = {}".format(repr(str( self.if_path       )))
                    out_str += space1[2]+"      of_path = {}".format(repr(str( self.of_path       )))
                    out_str += space1[2]+"      od_path = {}".format(repr(str( self.od_path       )))
                    out_str += space1[2]+"      of_stem = {}".format(repr(str( self.of_stem       )))
                    out_str += space1[2]+"      of_base = {}".format(repr(str( self.of_base       )))
                    out_str += space1[2]+" of_BAD_chans = {}".format(repr(str( self.of_BAD_chans  )))
                    out_str += space1[2]+" of_BAD_spans = {}".format(repr(str( self.of_BAD_spans  )))
                    out_str += space1[2]+"of_BAD_epochs = {}".format(repr(str( self.of_BAD_epochs )))
                    out_str += space1[2]+" of_BAD_comps = {}".format(repr(str( self.of_BAD_comps  )))
                    out_str += space1[2]+"      of_rand = {}".format(repr(str( self.of_rand       )))
                    out_str += space1[2]+"      of_data = {}".format(repr(str( self.of_data       )))
                    out_str += space1[2]+"      of_done = {}".format(repr(str( self.of_done       )))

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
            ## JOB BATCHES
            ## =============================================================================




            def JOB_000(
                    self,
                    ASK           = False,
                    cleanup       = False,
            ):
                """
                This JOB
                - reads raw data (BrainVision)
                - applies very basic pre-processing steps (filtering ETC)
                - produces basic plots that should be used for data QC
                - SHOULD end up with
                  - filtered data
                  - BAD channels interpolated [???]

                """
                temp_continue = "\nPress any (standard) key to continue... "

                self.info()
                if ASK & sys.stdout.isatty(): input(temp_continue)

                """
                Read RAW data
                """
                self.read_raw_data(
                    raw0    = "raw0",
                    preload = True,
                    verbose = None,
                )
                self.check_chans_number(
                    raw0 = "raw0",
                )
                """
                If present BAD channels SHOULD be loaded during the SECOND RUN
                """
                self.check_BAD_chans_file(
                    raw0 = "raw0",
                )
                self.add_actual_reference(
                    raw0          = "raw0",
                    ref_chans_OLD = self.BATCH.dataBase.setup["chans"]["refs"],
                )
                self.update_montage(
                    raw0    = "raw0",
                    montage = "standard_1005",
                )
                """
                PLOTS
                - ver000_very_raw
                """
                if sys.stdout.isatty(): plt.close("all")
                self.plot_channels_power_spectral_density__AND__plot_raw_data_timeseries(
                    raw0    = "raw0",
                    total   = True,
                    average = False,
                    exclude = False,
                    showFig = False,
                    suffFig = "ver000_very_raw",
                    saveFig = True,
                )
                if sys.stdout.isatty(): plt.close("all")

                """
                INITIAL PREPROCESSING
                - average reference projection
                - process events and annots
                - bandpass filter
                """
                self.average_reference_projection(
                    raw0          = "raw0",
                    ref_chans_NEW = "average",
                )
                self.process_events_and_annots(
                    raw0    = "raw0",
                    annots0 = "annots0",
                    events0 = "events0",
                    orig0   = "orig0",
                    desc0   = "desc0",
                    time0   = "time0",
                )
                ## currently BAD spans checkup is deprecated,
                ## instead in the following steps epoched data
                ## will be cleaned using ICA and amplitude thresholds...

                # self.BATCH.logger.info("checking for bad spans")
                # self.check_for_BAD_spans_file()

                self.bandpass_filter(
                    raw0       = "raw0",
                    l_freq     = self.BATCH.dataBase.setup["filt"]["l_freq"],
                    h_freq     = self.BATCH.dataBase.setup["filt"]["h_freq"],
                    fir_design = self.BATCH.dataBase.setup["filt"]["fir_design"],
                )

                """
                PLOTS
                - ver04_after_bandpass
                """
                if sys.stdout.isatty(): plt.close("all")

                self.plot_channels_power_spectral_density__AND__plot_raw_data_timeseries(
                    raw0    = "raw0",
                    total   = True,
                    average = False,
                    exclude = False,
                    showFig = False,
                    suffFig = "ver004_after_bandpass",
                    saveFig = True,
                )
                if sys.stdout.isatty(): plt.close("all")

                ## currently manual inspection of RAW (continous)
                ## data is deprecated
                ## instead, epoched data will be inspected
                ## in following steps and
                ## it will be cleaned using ICA and amplitude thresholds...
                ## ALSO: BAD spans data should not be loaded OR written

                # if sys.stdout.isatty(): plt.close("all")
                # self.BATCH.logger.info("plot RAW data timeseries for MANUAL INSPECTION")
                # self.plot_raw_data_timeseries(total=False,exclude=False,)
                # if ASK & sys.stdout.isatty(): input(temp_continue)
                # if sys.stdout.isatty(): plt.close("all")
                # self.BATCH.logger.info("save BAD spans after MANUAL INSPECTION ")
                # self.export_BAD_spans_info()

                """
                construct epochs (preload=False)
                - no rejections now
                """
                self.extract_metadata_for_events_acquired(
                    events0 = "events0",
                    meta0   = "meta0",
                    meta1   = "meta1",
                    time0   = "time0",
                    time1   = "time1",
                    desc0   = "desc0",
                    desc1   = "desc1",
                )
                self.construct_epochs(
                    raw0    = "raw0",
                    events0 = "events0",
                    epochs0 = "epochs0",
                    meta1   = "meta1",
                    time1   = "time1",
                    desc1   = "desc1",
                    reject  = None,
                    flat    = None,
                    preload = False,
                )

                """
                PLOTS
                - ver005_before_BAD_epochs_drop
                """
                if sys.stdout.isatty(): plt.close("all")
                self.plot_epochs_using_chan_BUNDLES_carpet(
                    epochs0 = "epochs0",
                    showFig = False,
                    suffFig = "ver005_before_BAD_epochs_drop",
                )
                if sys.stdout.isatty(): plt.close("all")

                """
                check BAD epochs file (as selected by USER)
                """
                self.check_BAD_epochs_file(
                    epochs0 = "epochs0",
                )

                """
                PLOTS
                - ver006_after_BAD_epochs_drop
                """
                if sys.stdout.isatty(): plt.close("all")
                self.plot_epochs_using_chan_BUNDLES_carpet(
                    epochs0 = "epochs0",
                    showFig = False,
                    suffFig = "ver006_after_BAD_epochs_drop",
                )
                if sys.stdout.isatty(): plt.close("all")

                """
                construct evoked
                """
                self.construct_evoked_WORD_SET(evoked0="evoked0",epochs0="epochs0")
                self.construct_evoked_WORD_LEN(evoked0="evoked0",epochs0="epochs0")

                if sys.stdout.isatty(): plt.close("all")
                self.plot_evoked_JOINT(
                    evoked0           = "evoked0",
                    quest0            = "word_set",
                    apply_projections = True,
                    interpolate_bads  = True,
                    showFig           = False,
                    suffFig           = "",
                )
                self.plot_evoked_JOINT(
                    evoked0           = "evoked0",
                    quest0            = "word_len",
                    apply_projections = True,
                    interpolate_bads  = True,
                    showFig           = False,
                    suffFig           = "",
                )
                if sys.stdout.isatty(): plt.close("all")

                self.extract_PEAKS_from_evoked_to_dataframe(
                    df0_peaks0 = "df0_peaks0",
                    evoked0    = "evoked0",
                    chans0     = self.BATCH.dataBase.setup["chans"]["bund1"]["B1"],
                    bunds0     = self.BATCH.dataBase.setup["chans"]["bund0"].items(),
                    timespans0 = list(self.BATCH.dataBase.setup["time"]["spans0"].values()),
                    stupid     = False,
                )
                self.evoked_to_dataframe(
                    evoked0      = "evoked0",
                    quest0       = "word_set",
                    df0_evoked0  = "df0_evoked0_word_set",
                )
                self.epochs_to_dataframe(
                    epochs0     = "epochs0",
                    events0     = "events0",
                    meta1       = 'meta1',
                    df0_epochs0 = "df0_epochs0",
                )
                if sys.stdout.isatty(): plt.close("all")
                chans1 = ["C3","C4","F3","F4","PO3","PO4","O1","O2",]
                chans2 = ["Fz","Cz","Pz","Oz","CPz","POz","FCz"]
                chans0 = chans2
                chans0 = chans0 + chans2
                chans0 = chans1
                chans0 = None
                bunds0 = self.BATCH.dataBase.setup["chans"]["bund0"]
                self.plot_evoked_COMPARE(
                    evoked0     = "evoked0",  # "evoked0" OR "evoked2" OR ...
                    quest0      = "word_set", # "word_set" OR "word_len"
                    chans0      = chans0,
                    bunds0      = bunds0,
                    showFig     = False,
                    suffFig     = "",
                    colors0     = self.BATCH.dataBase.setup["colors0"]["word_set"],
                    styles0     = self.BATCH.dataBase.setup["styles0"]["word_set"],
                    linestyles0 = self.BATCH.dataBase.setup["linestyles0"]["word_set"],
                    vlines7     = self.BATCH.dataBase.setup["time"]["vlines7"],
                    timespans0  = list(self.BATCH.dataBase.setup["time"]["spans0"].values()),
                    colors6     = self.BATCH.dataBase.setup["time"]["colors6"],
                    df0_peaks0  = "df0_peaks0",
                )
                self.plot_evoked_COMPARE(
                    evoked0     = "evoked0",  # "evoked0" OR "evoked2" OR ...
                    quest0      = "word_len", # "word_set" OR "word_len"
                    chans0      = chans0,
                    bunds0      = bunds0,
                    showFig     = False,
                    suffFig     = "",
                    colors0     = self.BATCH.dataBase.setup["colors0"]["word_len"],
                    styles0     = self.BATCH.dataBase.setup["styles0"]["word_len"],
                    linestyles0 = self.BATCH.dataBase.setup["linestyles0"]["word_len"],
                    vlines7     = self.BATCH.dataBase.setup["time"]["vlines7"],
                    timespans0  = list(self.BATCH.dataBase.setup["time"]["spans0"].values()),
                    colors6     = self.BATCH.dataBase.setup["time"]["colors6"],
                    df0_peaks0  = "df0_peaks0",
                )
                if sys.stdout.isatty(): plt.close("all")

                """
                ## MANUAL INSPECTION of EPOCHS

                self.inspect_epochs(
                    epochs0    = "epochs0",
                    excludeBAD = False,
                )
                self.export_BAD_epochs_info(
                    epochs0 = "epochs0",
                )
                """
                self.drop_BAD_epochs()


                """
                self.write_hkl()
                self.mark_DONE()
                """

                if cleanup:
                    self.data = OrderedDict()








            def JOB_001(self,ASK=True):
                pass









            def JOB_002(self,ASK=True):
                pass

                ### TODO FIXME ALSO PRODUCE STATS HERE FOR COMPARISON WITH CLEANED DATA



            def JOB_003(self,ASK=True):
                temp_continue = "\nPress any (standard) key to continue... "


                self.info()
                if ASK & sys.stdout.isatty(): input(temp_continue)


                self.run_ica(
                    ica0         = "ica0",
                    epochs0      = "epochs0",
                    n_components = 0.98,
                )


                if sys.stdout.isatty(): plt.close("all")
                self.inspect_components()
                ### if ASK & sys.stdout.isatty(): input(temp_continue)
                ### self.export_components(of_suff=".0.test")
                ### self.import_components(of_suff=".0.test")
                if sys.stdout.isatty(): plt.close("all")





                if sys.stdout.isatty(): plt.close("all")
                # if ASK & sys.stdout.isatty(): input(temp_continue)
                if sys.stdout.isatty(): plt.close("all")

                self.plot_epochs_drop_log()












                if sys.stdout.isatty(): plt.close("all")
                self.plot_component_properties()
                if sys.stdout.isatty(): plt.close("all")


                self.apply_projections_and_interpolate_bads(
                    ica0="ica0",
                    epochs0="epochs0",
                    epochs1="epochs1",
                    epochs2="epochs2",
                )


                if sys.stdout.isatty(): plt.close("all")
                self.plot_epochs_using_chan_BUNDLES_carpet(epochs0="epochs2")
                if sys.stdout.isatty(): plt.close("all")


                self.construct_evoked_WORD_SET(evoked0="evoked2",epochs0="epochs2")
                self.construct_evoked_WORD_LEN(evoked0="evoked2",epochs0="epochs2")


                if sys.stdout.isatty(): plt.close("all")
                self.plot_evoked_JOINT(evoked0="evoked2",quest0="word_set",)
                self.plot_evoked_JOINT(evoked0="evoked2",quest0="word_len",)
                if sys.stdout.isatty(): plt.close("all")

                if sys.stdout.isatty(): plt.close("all")
                chans0 = ["C3","C4","F3","F4","PO3","PO4",]
                chans0 = ["C3","C4","F3","F4","PO3","PO4","O1","O2",]

                self.plot_evoked_COMPARE(evoked0="evoked2",chans0=chans0,quest0="word_set",)
                self.plot_evoked_COMPARE(evoked0="evoked2",chans0=chans0,quest0="word_len",)
                if sys.stdout.isatty(): plt.close("all")

                ### self.export_evoked_as_dataframe(evoked0="evoked2",quest0="word_set",df0_name="dfEvoked2",)
                ### self.export_epochs_as_dataframe(epochs0="epochs2",events0="events0",df0_name="dfEpochs2",)

                self.write_hkl()


            ## =============================================================================
            ## EEG signal preprocessing functions go below this point
            ## =============================================================================

            ## TODO FIXME CONSIDER adding "if_extn" property and
            ## a case switch for loading variety of file types
            ## using just one function



            def read_raw_data(
                    self,
                    raw0,           # raw0    = "raw0",
                    preload = True, # preload = True,
                    verbose = None, # verbose = None,
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
                of_data = self.locs.of_data
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"if_path: {}"   .format(repr(str( if_path ))))
                self.BATCH.logger.info (space0[1]+"of_stem: {}"   .format(repr(str( of_stem ))))
                self.BATCH.logger.info (space0[1]+"of_base: {}"   .format(repr(str( of_base ))))
                self.BATCH.logger.info (space0[1]+"of_data: {}"   .format(repr(str( of_data ))))
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
                self.BATCH.logger.info (space0[1]+"data dictionary REGENERATION...")
                self.data = OrderedDict()

                self.BATCH.logger.info (space0[1]+"loading data...")
                self.data[raw0] = mne.io.read_raw_brainvision(
                    **ARGS,
                )

                self.BATCH.logger.info (space0[1]+"updating dataset description...")
                self.data[raw0].info["description"] = str(of_stem)
                self.BATCH.logger.info (space0[1]+"DONE...")




            def read_hkl(
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
                of_data = self.locs.of_data
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"if_path: {}"   .format(repr(str( if_path ))))
                self.BATCH.logger.info (space0[1]+"of_stem: {}"   .format(repr(str( of_stem ))))
                self.BATCH.logger.info (space0[1]+"of_base: {}"   .format(repr(str( of_base ))))
                self.BATCH.logger.info (space0[1]+"of_data: {}"   .format(repr(str( of_data ))))
                self.BATCH.logger.info (space0[1]+"EXE:     {}"   .format("hkl.load()"))
                ARGS = dict(
                    fileobj = if_path,
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.BATCH.logger.info (space0[1]+"ALL data is being overwritten")
                self.data = hkl.load(
                    **ARGS,
                )
                self.BATCH.logger.info (space0[1]+"DONE...")




            def check_chans_number(
                    self,
                    raw0,
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
                    raw0,
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




            def add_actual_reference(
                    self,
                    raw0,
                    ref_chans_OLD,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"adding reference channel...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0 ))))

                self.BATCH.logger.info (space0[1]+"adding actual (OLD) reference channel(s) to data...")
                self.BATCH.logger.info (space0[2]+"ref_chans_OLD: {}".format(ref_chans_OLD))
                self.BATCH.logger.info (space0[2]+"EXEC: {}"         .format("mne.add_reference_channels()"))
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
                self.BATCH.logger.info(space0[1]+"Everything seems to be WELL...")




            def update_montage(
                    self,
                    raw0    = "raw0",
                    montage = "standard_1005",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"updating montage...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0    ))))
                self.BATCH.logger.info (space0[1]+"montage: {}"   .format(repr(str( montage ))))

                self.BATCH.logger.info (space0[1]+"setting montage...")
                self.BATCH.logger.info (space0[2]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(str(raw0)),"set_montage"))
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
                self.BATCH.logger.info(space0[1]+"Everything seems to be WELL...")




            def average_reference_projection(
                    self,
                    raw0,          # raw0          = "raw0",
                    ref_chans_NEW, # ref_chans_NEW = "average",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"adding reference projection...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self ))))
                self.BATCH.logger.info (space0[1]+"raw0: {}"      .format(repr(str( raw0 ))))

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
                    raw0,
                    annots0,
                    events0,
                    orig0,
                    desc0,
                    time0,
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




            def check_for_BAD_spans_file(
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
                    raw0,
                    l_freq,
                    h_freq,
                    fir_design,

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
                    l_freq     = l_freq,
                    h_freq     = h_freq,
                    fir_design = fir_design,
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




            def plot_channels_power_spectral_density__AND__plot_raw_data_timeseries(
                    self,
                    raw0,
                    total   = True,
                    average = False,
                    exclude = False,
                    showFig = False,
                    suffFig = "",
                    saveFig = True,
            ):
                self.plot_channels_power_spectral_density(
                    raw0    = raw0,
                    average = average,
                    exclude = exclude,
                    showFig = showFig,
                    suffFig = suffFig,
                    saveFig = saveFig,
                )
                self.plot_raw_data_timeseries(
                    raw0    = raw0,
                    total   = total,
                    exclude = exclude,
                    showFig = showFig,
                    suffFig = suffFig,
                    saveFig = saveFig,
                )




            def plot_channels_power_spectral_density(
                    self,
                    raw0,
                    average = False,
                    exclude = False,
                    showFig = False,
                    yLimVal = [-80,40],
                    suffFig = "",
                    saveFig = True,
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
                of_suff = ".".join([of_suff,suffFig])
                of_suff = ".".join([of_suff,"png"])

                EXCLUDE = self.data[raw0].info["bads"] if exclude else []
                picks   = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=EXCLUDE,)

                self.BATCH.logger.info (space0[1]+"EXEC: {}[{}].{}".format(".".join(self.INSP),repr(raw0),"plot_psd"))
                ARGS = dict(
                    show    = False,
                    fmin    =   0,
                    fmax    =  60,
                    picks   = picks,
                    average = average,
                    proj    = True,
                    # xscale  = "log",
                )
                self.BATCH.logger.info(
                    space0[1]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                fig = self.data[raw0].plot_psd(
                    **ARGS,
                )
                fig.axes[0].set_ylim(yLimVal)
                fig.set_size_inches(16,8)
                title_old = fig.axes[0].get_title()
                title_new = "{} {} {} ({})\n{} {}".format(
                    title_old,
                    self.locs.of_stem,
                    raw0,
                    suffFig,
                    "chansAvg" if average else "chansSep",
                    "exclBAD"  if exclude else "inclBAD",
                )
                fig.axes[0].set(title=title_new)
                # plt.tight_layout(pad=.5)
                if showFig: (fig or plt).show()
                if saveFig:
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    fig.savefig(of_name, dpi=fig.dpi,)




            def plot_raw_data_timeseries(
                    self,
                    raw0,
                    total   = False,
                    exclude = False,
                    showFig = False,
                    suffFig = "",
                    saveFig = True,
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
                of_suff = ".".join([of_suff,suffFig])
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
                title_new = "{} {} {} ({})\n{} {}".format(
                    title_old,
                    self.locs.of_stem,
                    raw0,
                    suffFig,
                    "fullSig" if total   else "someSig",
                    "exclBAD" if exclude else "inclBAD",
                )
                ## TODO FIXME no need to update title
                # fig.axes[0].set(title=title_new)

                # plt.tight_layout(pad=.5)
                if showFig: (fig or plt).show()
                if saveFig:
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
                    events0,
                    meta0,
                    meta1,
                    time0,
                    time1,
                    desc0,
                    desc1,
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

                df0 = self.data[events0][meta0].copy(deep=True)[
                    (self.data[events0][meta0]["CODE"] > 100) &
                    (self.data[events0][meta0]["CODE"] < 300)
                ]
                self.BATCH.logger.info (space0[2]+"computing onset diff column for {}...".format(repr(str(meta1))))
                self.BATCH.logger.info (space0[2]+"... this (re)computation is requided after...")
                self.BATCH.logger.info (space0[2]+"... removing events not related to stimuli were dropped...")
                # df0.loc[:,"DIFF"] = df0["ONSET"].diff()
                df0["DIFF"] = df0["ONSET"].diff()
                df0["DIFF"].fillna(0,inplace=True)

                self.BATCH.logger.info (space0[1]+"merging event details in {} with extra info from SETUP...".format(repr(str( meta1 ))))
                self.BATCH.logger.info (space0[2]+"these will be kept in pandas DataFrame...")
                self.BATCH.logger.info (space0[2]+"under meta1: {} key ".format(repr(str( meta1 ))))
                df0 = pd.merge(
                    left      = df0,
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
                self.BATCH.logger.info (space0[2]+"adding extra columns (STEM,SUB,SES,TASK,RUN) to {}...".format(repr(str( meta1 ))))
                # df0["STEM"] = str(self.locs.of_stem)
                df0["SUB"]  = self.sub
                df0["SES"]  = self.ses
                df0["TASK"] = self.task
                df0["RUN"]  = self.run
                self.BATCH.logger.info (space0[1]+"exporting event onset information from {}...".format(repr(str(meta1))))
                self.BATCH.logger.info (space0[2]+"these will be kept under time1: {} key..."   .format(repr(str(time1))))

                self.data[events0][time1] = df0[["ONSET","DURATION","CODE",]].to_numpy()


                self.BATCH.logger.info (space0[1]+"producing exclusively stimuli-related events descriptions...")
                self.BATCH.logger.info (space0[2]+"these will be kept under desc1: {} key...".format(repr(str(desc1))))
                self.data[events0][desc1] = OrderedDict()
                for (key0, val0) in self.data[events0][desc0].items():
                    if (val0 > 100) & (val0 < 300):
                        self.data[events0][desc1][key0] = val0

                self.BATCH.logger.info (space0[1]+"checking assertions...")
                # assert len(df0) == 80       , "PROBLEM: Please fix meta1"
                assert len(df0) in [80,160,240,320]       , "PROBLEM: Please fix meta1"

                self.data[events0][meta1] = dc(df0)
                # assert len(self.data[events0][desc1]) == 80       , "PROBLEM: Please fix desc1"
                # assert self.data[events0][time1].shape == (80, 3) , "PROBLEM: Please fix time1"
                self.BATCH.logger.info (space0[1]+"all assertions were met... GREAT!")




            def construct_epochs(
                    self,
                    raw0,
                    events0,
                    epochs0,
                    meta1,
                    time1,
                    desc1,
                    reject  = None,
                    flat    = None,
                    preload = False,
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

                if reject == "setup":
                    reject = self.BATCH.dataBase.setup["params"]["reject"]

                if flat == "setup":
                    flat = self.BATCH.dataBase.setup["params"]["flat"]

                self.BATCH.logger.info (space0[1]+"reject: {}".format(repr(str( reject ))))
                self.BATCH.logger.info (space0[1]+"flat: {}"  .format(repr(str( flat   ))))

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
                    reject_by_annotation = True,
                    reject   = reject,
                    flat     = flat,
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
                    epochs0,
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
                self.BATCH.logger.debug(space0[2]+"looking for of_BAD_epochs: {}".format(repr(str(of_BAD_epochs))))
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
                    epochs0,     # epochs0    = "epochs0",
                    excludeBAD = True,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"inspecting epochs (possibly MARKING BAD)")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self       ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0    ))))
                self.BATCH.logger.info (space0[1]+"excludeBAD: {}".format(repr(str( excludeBAD ))))

                EXCLUDE = self.data[epochs0].info["bads"] if excludeBAD else []
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
                    reject  = "setup",
                    flat    = "setup",

            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"dropping epochs marked as BAD")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                if reject == "setup":
                    reject = self.BATCH.dataBase.setup["params"]["reject"]

                if flat == "setup":
                    flat = self.BATCH.dataBase.setup["params"]["flat"]

                self.BATCH.logger.info (space0[1]+"reject: {}".format(repr(str( reject ))))
                self.BATCH.logger.info (space0[1]+"flat: {}"  .format(repr(str( flat   ))))

                ARGS = dict(
                    reject   = reject,
                    flat     = flat,
                )
                self.BATCH.logger.info(
                    space0[2]+"ARGS:\n{}".format(
                        str_dict(ARGS,"   ARGS",max_level=0,max_len=77,)
                    )
                )
                self.data[epochs0].drop_bad(
                      **ARGS,
                )
                self.BATCH.logger.info (space0[1]+"DONE with epochs dropping...")
                ## TODO FIXME check if this truely operates inplace (in-place)




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
                    showFig = False,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting epoch drop log histogram...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                fig = self.data[epochs0].plot_drop_log(
                    show = False,
                )
                fig.set_size_inches(16,8)

                title_old = fig.axes[0].get_title()
                title_new = "{}\n{} {}".format(
                    self.locs.of_stem,
                    epochs0,
                    title_old,
                )
                fig.axes[0].set(title=title_new)
                if showFig: (fig or plt).show()
                of_suff = ""
                of_suff = ".".join([of_suff,str(whoami()),epochs0])
                of_suff = ".".join([of_suff,"0"])
                of_suff = ".".join([of_suff,"png"])
                of_name = self.locs.of_base.with_suffix(of_suff)
                self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                fig.savefig(of_name, dpi=fig.dpi,)




            def plot_epochs_using_chan_BUNDLES_carpet(
                    self,
                    epochs0 = "epochs0",
                    showFig = False,
                    suffFig = "",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting epochs using channel BUNDLES...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                elec_idx = OrderedDict()
                dict0 = dc(self.BATCH.dataBase.setup["chans"]["bund0"])
                dict1 = dc(self.BATCH.dataBase.setup["chans"]["bund1"])

                assert not bool(
                    set(dict0.keys()) & set(dict1.keys())
                ), "keys should be unique for channel bundles"

                bundles = {
                    **dict0,
                    **dict1,
                }
                for key,val in bundles.items():
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
                        fig.set_size_inches(24,16)
                        title_old = fig.axes[0].get_title()
                        title_new = "{}\n{} ({}) {}".format(
                            self.locs.of_stem,
                            epochs0,
                            suffFig,
                            title_old,
                        )
                        fig.axes[0].set(title=title_new)
                        title_bndl = title_old.partition(" ")[0]
                        if showFig: (fig or plt).show()
                        of_suff = ""
                        of_suff = ".".join([of_suff,str(whoami())])
                        of_suff = ".".join([of_suff,"{:03d}".format(jj)])
                        of_suff = ".".join([of_suff,str(combine)])
                        of_suff = ".".join([of_suff,"{:03d}".format(ii)])
                        of_suff = ".".join([of_suff,str(title_bndl)])
                        of_suff = ".".join([of_suff,epochs0])
                        of_suff = ".".join([of_suff,suffFig] if suffFig else [of_suff])
                        of_suff = ".".join([of_suff,"png"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                        fig.savefig(of_name, dpi=fig.dpi,)




            def construct_evoked_WORD_SET(
                    self,
                    evoked0, # evoked0 = "evoked0",
                    epochs0, # epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"constructing evoked...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"   .format(repr(str( evoked0 ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                self.data[evoked0]             = OrderedDict()
                self.data[evoked0]["word_set"] = OrderedDict()
                self.data[evoked0]["word_set"]["all_in"] = self.data[epochs0].average()
                for ii,(key,val) in enumerate(self.BATCH.dataBase.setup["queries"]["word_set"].items()):
                    self.BATCH.logger.debug(space0[1]+"{}: {}"   .format(repr(str(key)),repr(str(val))))
                    self.data[evoked0]["word_set"][key]  = self.data[epochs0][val].average()
                self.BATCH.logger.info (space0[1]+"DONE...")




            def construct_evoked_WORD_LEN(
                    self,
                    evoked0, # evoked0 = "evoked0",
                    epochs0, # epochs0 = "epochs0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"constructing evoked with respect to word length")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"   .format(repr(str( evoked0 ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))
                # self.data[evoked0]             = OrderedDict()
                self.data[evoked0]["word_len"] = OrderedDict()
                query = 'LEN == {}'
                for n_letters in sorted( self.data[epochs0].metadata["LEN"].unique()):
                    self.BATCH.logger.debug(space0[1]+"n_letters: {}"   .format(repr(str(n_letters))))
                    self.data[evoked0]["word_len"][str(n_letters)] = self.data[epochs0][query.format(n_letters)].average()




            def extract_PEAKS_from_evoked_to_dataframe(
                    self,
                    df0_peaks0,     # df0_peaks0 = "df0_peaks0",
                    evoked0,        # evoked0    = "evoked2",
                    chans0,         # chans0     = self.BATCH.dataBase.setup["chans"]["bund1"]["B1"],
                    bunds0,         # bunds0     = self.BATCH.dataBase.setup["chans"]["bund0"].items(),
                    timespans0,     # timespans0 = list(self.BATCH.dataBase.setup["time"]["spans0"].values()),
                    stupid = False, # stupid     = False,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"extracting peaks from evoked for some data channels accross conditions")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self        ))))
                self.BATCH.logger.info (space0[1]+"df0_peaks0: {}"    .format(repr(str( df0_peaks0      ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"   .format(repr(str( evoked0     ))))
                self.BATCH.logger.info (space0[1]+"chans0: {}"    .format(repr(str( chans0      ))))
                self.BATCH.logger.info (space0[1]+"bunds0: {}"    .format(repr(str( bunds0      ))))
                self.BATCH.logger.info (space0[1]+"timespans0: {}".format(repr(str( timespans0  ))))
                self.data[df0_peaks0] = None
                df0 = pd.DataFrame(
                    [],
                    columns=["evoked0","quest0","cond0","chan0","tmin0","tmax0","mode0","chanX","latX","valX"],
                )
                for quest0 in self.data[evoked0].keys():
                    self.BATCH.logger.info (space0[1]+"PROC quest0: {}".format(repr(str( quest0 ))))
                    for idx0,(cond0,val0) in enumerate(self.data[evoked0][quest0].items()):
                        self.BATCH.logger.info (space0[2]+"cond0: {}".format(repr(str( cond0 ))))
                        for chan0 in chans0:
                            self.BATCH.logger.info (space0[2]+"chan0: {}".format(repr(str( chan0 ))))
                            for timespan0 in timespans0:
                                tmin0 = timespan0[0]
                                tmax0 = timespan0[1]
                                for mode0 in ["pos","neg"]:
                                    smin0,smax0 = val0.time_as_index([tmin0,tmax0])
                                    chan_data = val0.copy().pick(chan0).data[0,smin0:smax0]
                                    if mode0 == "neg": chan_data = -chan_data
                                    chan_peak,_ = find_peaks(
                                        x            = chan_data,
                                        height       = None,
                                        threshold    = None,
                                        distance     = 40000,
                                        prominence   = None,
                                        width        = None,
                                        wlen         = None,
                                        rel_height   = 0.5,
                                        plateau_size = None,
                                    )
                                    chanX = chan0
                                    latX  = val0.times[chan_peak+smin0]
                                    valX  = chan_data[chan_peak]
                                    latX  = latX[0] if latX.size > 0 else np.nan
                                    valX  = valX[0] if valX.size > 0 else np.nan
                                    if mode0 == "neg": valX = -valX
                                    if stupid:
                                        chanX,latX,valX = val0.copy().pick(chan0).get_peak(
                                            ch_type          = "eeg",
                                            tmin             = tmin0,
                                            tmax             = tmax0,
                                            mode             = mode0,
                                            return_amplitude = True,
                                        )

                                    df1 = pd.DataFrame(
                                        [[evoked0,quest0,cond0,chan0,tmin0,tmax0,mode0,chanX,latX,valX]],
                                        columns=["evoked0","quest0","cond0","chan0","tmin0","tmax0","mode0","chanX","latX","valX"],
                                    )
                                    df1["valX"] = df1["valX"]*1e6
                                    # df1["STEM"] = str(self.locs.of_stem)
                                    df1["SUB"]  = self.sub
                                    df1["SES"]  = self.ses
                                    df1["TASK"] = self.task
                                    df1["RUN"]  = self.run
                                    df0 = df0.append(df1)


                        inst0     = dc(val0)
                        bunds0idx = OrderedDict()
                        for key2,val2 in bunds0:
                            self.BATCH.logger.info (space0[2]+"bunds0: {}".format(repr(str( bunds0 ))))
                            self.BATCH.logger.info (space0[2]+"bunds0 key2: {}".format(repr(str( key2 ))))
                            self.BATCH.logger.info (space0[2]+"bunds0 val2: {}".format(repr(str( val2 ))))
                            bunds0idx[key2] = mne.pick_channels(inst0.info['ch_names'], val2 )

                        ## TODO FIXME CONSIDER avoiding code repetition and use combine_channels for every single channel
                        inst1 = mne.channels.combine_channels(
                            inst   = inst0,
                            groups = bunds0idx,
                            method = "mean",
                        )
                        for chan0 in inst1.info["ch_names"]:
                            self.BATCH.logger.info (space0[2]+"chan0: {}".format(repr(str( chan0 ))))
                            for timespan0 in timespans0:
                                tmin0 = timespan0[0]
                                tmax0 = timespan0[1]
                                for mode0 in ["pos","neg"]:
                                    smin0,smax0 = inst1.time_as_index([tmin0,tmax0])
                                    chan_data = inst1.copy().pick(chan0).data[0,smin0:smax0]
                                    if mode0 == "neg": chan_data = -chan_data
                                    chan_peak,_ = find_peaks(
                                        x            = chan_data,
                                        height       = None,
                                        threshold    = None,
                                        distance     = 40000,
                                        prominence   = None,
                                        width        = None,
                                        wlen         = None,
                                        rel_height   = 0.5,
                                        plateau_size = None,
                                    )
                                    chanX = chan0
                                    latX  = inst1.times[chan_peak+smin0]
                                    valX  = chan_data[chan_peak]
                                    latX  = latX[0] if latX.size > 0 else np.nan
                                    valX  = valX[0] if valX.size > 0 else np.nan
                                    if mode0 == "neg": valX = -valX
                                    if stupid:
                                        chanX,latX,valX = inst1.copy().pick(chan0).get_peak(
                                            ch_type          = "eeg",
                                            tmin             = tmin0,
                                            tmax             = tmax0,
                                            mode             = mode0,
                                            return_amplitude = True,
                                        )

                                    df1 = pd.DataFrame(
                                        [[evoked0,quest0,cond0,chan0,tmin0,tmax0,mode0,chanX,latX,valX]],
                                        columns=["evoked0","quest0","cond0","chan0","tmin0","tmax0","mode0","chanX","latX","valX"],
                                    )
                                    df1["valX"] = df1["valX"]*1e6
                                    # df1["STEM"] = str(self.locs.of_stem)
                                    df1["SUB"]  = self.sub
                                    df1["SES"]  = self.ses
                                    df1["TASK"] = self.task
                                    df1["RUN"]  = self.run
                                    df0 = df0.append(df1)

                di0 = self.BATCH.dataBase.setup["chans"]["bund0"]
                di1 = OrderedDict()
                for key0,val0 in di0.items():
                    for item in val0:
                        di1[item] = key0

                df0["CHAN_BUND"] = df0["chan0"].apply(lambda x: di1[x] if x in di1 else None)
                self.data[df0_peaks0] = dc(df0)




            def plot_evoked_JOINT(
                    self,
                    evoked0,                  # evoked0           = "evoked0",  # OR "evoked2" OR ...
                    quest0,                   # quest0            = "word_set", # OR "word_len"
                    apply_projections = True, # apply_projections = True,
                    interpolate_bads  = True, # interpolate_bads  = True,
                    showFig           = False,# showFig           = False,
                    suffFig           = "",   # suffFig           = "",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting evoked")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"   .format(repr(str( evoked0 ))))

                spatial_colors = False
                spatial_colors = True
                ## Time points for which topomaps will be displayed
                times  = "auto"
                times  = list()
                times += [-0.200,-0.100,0,]
                times += self.BATCH.dataBase.setup["time"]["means0"].values()
                # times += list(sum(self.BATCH.dataBase.setup["time"]["spans0"].values(), ()))
                times = sorted(times)
                for ii,(key,evoked) in enumerate(self.data[evoked0][quest0].items()):
                    title = ""
                    title += "EEG ({}) ".format( evoked.info["nchan"] - len(evoked.info["bads"]) )
                    title += str(self.locs.of_stem)
                    evoked = evoked.copy()
                    if apply_projections:
                        evoked = evoked.apply_proj()

                    if interpolate_bads:
                        evoked =  evoked.interpolate_bads(
                            reset_bads = True,
                            mode       = "accurate",
                        )

                    fig = evoked.plot_joint(
                        show    = False,
                        title   = title + "\n{} {} ({})".format(key,evoked0,suffFig),
                        times   = times,
                        ts_args = dict(
                            time_unit      = "s",
                            ylim           = dict(eeg=[-15, 15]),
                            spatial_colors = spatial_colors,
                            gfp            = True,
                        ),
                        topomap_args = dict(
                            cmap     = "Spectral_r",
                            outlines = "skirt",
                        ),
                    )
                    fig.set_size_inches(16,8)
                    if showFig: (fig or plt).show()
                    of_suff = ""
                    of_suff = ".".join([of_suff,str(whoami())])
                    of_suff = ".".join([of_suff,str(quest0)])
                    of_suff = ".".join([of_suff,"{:03d}".format(ii)])
                    of_suff = ".".join([of_suff,str(key)])
                    of_suff = ".".join([of_suff,evoked0])
                    of_suff = ".".join([of_suff,suffFig] if suffFig else [of_suff])
                    of_suff = ".".join([of_suff,"png"])
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    fig.savefig(of_name, dpi=fig.dpi,)




            def plot_evoked_COMPARE(
                    self,
                    evoked0,             # evoked0     = "evoked0",  # OR "evoked2" OR ...
                    quest0,              # quest0      = "word_set", # OR "word_len"
                    chans0      = None,  # chans0      = self.BATCH.dataBase.setup["chans"]["bund1"]["B1"],
                    bunds0      = None,  # bunds0      = self.BATCH.dataBase.setup["chans"]["bund0"],
                    showFig     = False, # showFig     = False,
                    saveFig     = True,  # saveFig     = True,
                    suffFig     = "",    # suffFig     = "",
                    colors0     = None,  # colors0     = self.BATCH.dataBase.setup["colors0"]["word_set"],
                    styles0     = None,  # styles0     = self.BATCH.dataBase.setup["styles0"]["word_set"],
                    linestyles0 = None,  # linestyles0 = self.BATCH.dataBase.setup["linestyles0"]["word_set"],
                    vlines7     = [0],   # vlines7     = self.BATCH.dataBase.setup["time"]["vlines7"],
                    timespans0  = None,  # timespans0  = list(self.BATCH.dataBase.setup["time"]["spans0"].values()),
                    colors6     = None,  # colors6     = self.BATCH.dataBase.setup["time"]["colors6"],
                    df0_peaks0  = None,  # df0_peaks0  = "df0_peaks0",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"plotting evoked for some data channels accross conditions")
                self.BATCH.logger.info (space0[1]+"processing: {}" .format(repr(str( self        ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"    .format(repr(str( evoked0     ))))
                self.BATCH.logger.info (space0[1]+"quest0: {}" .format(repr(str( quest0 ))))
                self.BATCH.logger.info (space0[1]+"chans0: {}" .format(repr(str( chans0  ))))

                if (chans0 is not None) & (bunds0 is not None):
                    assert 0 == len(set(bunds0.keys()).intersection(set(chans0))), "PROBLEM: channel names (chans0) overlap with channel bundle names (bunds0)"

                picks0 = OrderedDict()
                if (chans0 is not None):
                    for chan0      in chans0:         picks0[chan0] = chan0
                if (bunds0 is not None):
                    for key0,bund0 in bunds0.items(): picks0[key0] = bund0

                assert 0 < len(picks0), "PROBLEM: chans0 OR bunds0 should not be empty"

                combine  = "gfp"
                combine  = "mean"

                for idx0,(key0,pick0) in enumerate(picks0.items()):
                    title = ""
                    title = " ".join([title,"Evoked: {} ".format( self.locs.of_stem )])
                    title = " ".join([title,"({})".format(evoked0)])
                    title = " ".join([title,"{}".format(key0)])
                    title = " ".join([title,"[{}]".format(pick0) if isinstance(pick0, str) else "{}".format(pick0)])
                    ## Typically a list of figures is created with "plot_compare_evokeds()"
                    ## Here (for each iteration) we produce list with exactly one item
                    ## Hence below we use "figs[0]"
                    figs = mne.viz.plot_compare_evokeds(
                        evokeds      = self.data[evoked0][quest0],
                        picks        = pick0,
                        ci           = 0.95,
                        ylim         = dict(eeg=[-12,12]),
                        invert_y     = True,
                        title        = title,
                        show         = False,
                        combine      = combine,
                        colors       = colors0,
                        styles       = styles0,
                        linestyles   = linestyles0,
                        vlines       = vlines7,
                        show_sensors = True,
                    )
                    figs[0].set_size_inches(16,8)
                    labels0 = figs[0].axes[0].get_xticklabels()
                    for label0 in labels0:
                        label0.set_rotation(45)

                    if (timespans0 is not None) & (colors6 is not None):
                        ## convert timespans (t0,t1) to start and duration
                        params0 = [ [time0[0],time0[1]-time0[0]] for time0 in timespans0 ]
                        for ii,param0 in enumerate(params0):
                            rect0 = matplotlib.patches.Rectangle(
                                [param0[0],-12],
                                param0[1],
                                24,
                                angle=0.0,
                                color=colors6[ii],
                                alpha=0.3,)
                            figs[0].axes[0].add_patch(rect0)

                    if df0_peaks0 is not None:
                        df0 = dc(self.data[df0_peaks0])
                        ## TODO FIXME CONSIDER if sorting is necessary here to reflect figs[0] order of stuff
                        arPos = df0[ True
                                     & (df0["evoked0"] == evoked0)
                                     & (df0["quest0"]  == quest0)
                                     & (df0["chanX"]   == key0)
                                     & (df0["mode0"]   == "pos")
                        ][["latX","valX"]].to_numpy(copy=True,)
                        arNeg = df0[ True
                                     & (df0["evoked0"] == evoked0)
                                     & (df0["quest0"]  == quest0)
                                     & (df0["chanX"]   == key0)
                                     & (df0["mode0"]   == "neg")
                        ][["latX","valX"]].to_numpy(copy=True,)
                        facecolors0 = list(colors0.values())
                        facecolors0 = list(itertools.chain.from_iterable(itertools.repeat(col0, len(colors6)) for col0 in facecolors0))
                        edgecolors0 = colors6*len(colors0)
                        temp_dist = 0.0 # 0.2
                        figs[0].axes[0].scatter(arPos.T[0],arPos.T[1]+temp_dist,s=48,facecolors=facecolors0,edgecolors=edgecolors0,marker="^",zorder=1200)
                        figs[0].axes[0].scatter(arNeg.T[0],arNeg.T[1]-temp_dist,s=48,facecolors=facecolors0,edgecolors=edgecolors0,marker="v",zorder=1200)

                    if showFig: (figs[0] or plt).show()
                    if saveFig:
                        of_suff = ""
                        of_suff = ".".join([of_suff,str(whoami())])
                        of_suff = ".".join([of_suff,str(quest0)])
                        of_suff = ".".join([of_suff,"{:03d}".format(idx0)])
                        of_suff = ".".join([of_suff,"CHAN" if isinstance(pick0, str) else "BUND"])
                        # of_suff = ".".join([of_suff,str(pick0).replace(" ","+",).replace("[","",).replace("]","",)])
                        # of_suff = ".".join([of_suff,str(pick0) if isinstance(pick0, str) else "+".join(pick0) ])
                        of_suff = ".".join([of_suff,str(key0)])
                        of_suff = ".".join([of_suff,evoked0])
                        of_suff = ".".join([of_suff,suffFig] if suffFig else [of_suff])
                        of_suff = ".".join([of_suff,"png"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                        figs[0].savefig(of_name, dpi=figs[0].dpi,)




            def run_ica(
                    self,
                    ica0         = "ica0",
                    epochs0      = "epochs0",
                    n_components = 0.98,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"running ICA...")
                self.BATCH.logger.info (space0[1]+"processing: {}"  .format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"        .format(repr(str( ica0    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"     .format(repr(str( epochs0 ))))
                self.BATCH.logger.info (space0[1]+"n_components: {}".format(repr(str( n_components ))))

                random_states = list()
                of_rand       = self.locs.of_rand
                self.BATCH.logger.info (space0[1]+"looking for of_rand: " + str(of_rand))
                if os.path.exists(of_rand):
                    self.BATCH.logger.info(space0[1]+"found random states file...")
                    with open(of_rand) as fh:
                        for line in fh:
                            line = line.split('#',1,)[0].strip()
                            if line:
                                random_states.append(int(line))

                else:
                    self.BATCH.logger.info(space0[1]+"random states file NOT found...")
                    random_states = [0]


                ## TODO FIXME
                # run_ica('infomax', fit_params=dict(extended=True))

                time_T0 = time.time()
                self.data[ica0] = mne.preprocessing.ica.ICA(
                    n_components       = n_components,
                    # n_components       = 50,
                    # n_pca_components   = 50,
                    # max_pca_components = 50,
                    method             = 'fastica',
                    max_iter           = 3200,
                    # noise_cov          = noise_cov,
                    random_state       = random_states[0],
                ).fit(
                    self.data[epochs0],
                )
                time_T1 = time.time()
                time_D1 = time_T1-time_T0
                self.BATCH.logger.info (space0[1]+"time elapsed: " + hf.format_timespan( time_D1 ))




            def inspect_components(
                    self,
                    ica0    = "ica0",
                    epochs0 = "epochs0",
                    showFig = False,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"inspecting ICs")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"      .format(repr(str( ica0    ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0 ))))

                title = "ICA Componentts\n"
                title += str(self.locs.of_stem)
                # TODO FIXME This can also output a single figure object
                # (not necesarily a list of figs)
                figs = self.data[ica0].plot_components(
                    inst   = self.data[epochs0],
                    title  = title,
                    show   = False,
                )
                for ii,fig in enumerate(figs):
                    fig.set_size_inches(16,16)
                    if showFig: (fig or plt).show()
                    of_suff = ""
                    of_suff = ".".join([of_suff,str(whoami()),epochs0,ica0])
                    of_suff = ".".join([of_suff,"{:03d}".format(ii)])
                    of_suff = ".".join([of_suff,"png"])
                    of_name = self.locs.of_base.with_suffix(of_suff)
                    self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                    fig.savefig(of_name, dpi=fig.dpi,)




            def import_components(
                    self,
                    ica0     = "ica0",
                    of_suff = "",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"importing ICs")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"      .format(repr(str( ica0    ))))
                self.BATCH.logger.info (space0[1]+"of_suff: {}"   .format(repr(str( of_suff ))))

                #of_BAD_comps = self.locs.of_BAD_comps.with_suffix(of_suff)
                of_BAD_comps = pathlib.Path(str(self.locs.of_BAD_comps)+of_suff)

                bad_comps = list()
                self.BATCH.logger.debug(space0[2]+"looking for of_BAD_comps: {}".format(repr(str(of_BAD_comps))))
                if os.path.exists(of_BAD_comps):
                    self.BATCH.logger.info(space0[2]+"found of_BAD_comps file...")
                    with open(of_BAD_comps) as fh:
                        for line in fh:
                            line = line.split('#',1,)[0].strip()
                            if line:
                                bad_comps.append(int(line))

                self.BATCH.logger.info (space0[2]+"bad_comps: {}".format(repr(str(bad_comps))))
                if bad_comps:
                    self.BATCH.logger.info(space0[1]+"adding BAD components informtion to data")
                    self.data[ica0].exclude = bad_comps

                else:
                    self.BATCH.logger.info(space0[1]+"no BAD components information was added to data")




            def export_components(
                    self,
                    ica0    = "ica0",
                    of_suff = "",
                    default = True,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting ICs...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self    ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"      .format(repr(str( ica0    ))))
                self.BATCH.logger.info (space0[1]+"of_suff: {}"   .format(repr(str( of_suff ))))
                self.BATCH.logger.info (space0[1]+"default: {}"   .format(repr(str( default ))))

                of_BAD_comps0 = self.locs.of_BAD_comps
                of_BAD_comps1 = pathlib.Path(str(self.locs.of_BAD_comps)+of_suff)

                self.BATCH.logger.info (space0[1]+"TGT of_BAD_comps0: {}"  .format(repr(str( of_BAD_comps0   ))))
                self.BATCH.logger.info (space0[1]+"TGT of_BAD_comps1: {}"  .format(repr(str( of_BAD_comps1   ))))

                exclude = dc(self.data[ica0].exclude)
                if exclude:

                    assert default | bool(of_suff), "PROBLEM: exporting to nowhere?!"

                    if default:
                        self.BATCH.logger.info (space0[1]+"writing BAD components informtion to file")
                        self.BATCH.logger.info (space0[1]+"using of_BAD_comps0: {}".format(repr(str( of_BAD_comps0 ))))
                        with open(of_BAD_comps0, 'w') as fh:
                            for item in exclude:
                                fh.write("{} # USER SELECTED BAD IC\n".format(item))

                    if of_suff:
                        self.BATCH.logger.info (space0[1]+"writing BAD components informtion to file")
                        self.BATCH.logger.info (space0[1]+"using of_BAD_comps1: {}".format(repr(str( of_BAD_comps1 ))))
                        with open(of_BAD_comps1, 'w') as fh:
                            for item in exclude:
                                fh.write("{} # USER SELECTED BAD IC\n".format(item))

                else:
                    self.BATCH.logger.info (space0[1]+"NOT writing BAD components informtion to file")
                    self.BATCH.logger.info (space0[1]+"SKIPPED because no components were selected")

                self.BATCH.logger.info (space0[1]+"DONE")




            def plot_component_properties(
                    self,
                    ica0     = "ica0",
                    epochs0  = "epochs0",
                    rejected = True,
                    saveFig  = True,
                    showFig  = False,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))

                self.BATCH.logger.info (space0[1]+"plotting ICs...")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self     ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"      .format(repr(str( ica0     ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0  ))))
                self.BATCH.logger.info (space0[1]+"rejected: {}"  .format(repr(str( rejected ))))
                self.BATCH.logger.info (space0[1]+"saveFig: {}"   .format(repr(str( saveFig  ))))

                exclude = sorted(self.data[ica0].exclude)
                # include = [item for item in list(range(self.data[ica0].n_components_)) if not item in exclude]
                # picks = exclude + include
                # picks = exclude if rejected else include
                # picks = None
                picks = list(range(self.data[ica0].n_components_))

                ## TODO FIXME add some logic here (or try+catch)
                ## for more elegant solution to dir handling
                # od_name = self.locs.od_path/"ICs"
                od_name = self.locs.of_base.with_suffix(".ICs")


                os.makedirs(od_name,mode=0o700,exist_ok=True,)
                shutil.rmtree(od_name)
                os.makedirs(od_name,mode=0o700,exist_ok=True,)

                if picks:
                    figs = self.data[ica0].plot_properties(
                        inst  = self.data[epochs0],
                        picks = picks,
                        show  = False,
                    )
                    for ii,fig in enumerate(figs):
                        STATUS = "EXC" if (ii in exclude) else "INC"
                        fig.set_size_inches(16,16)
                        title_old = fig.axes[0].get_title()
                        title_new = "{}\n{} {} {} [{:03d}] {}".format(
                            self.locs.of_stem,
                            ica0,
                            epochs0,
                            title_old,
                            ii,
                            STATUS,
                        )
                        fig.axes[0].set(title=title_new)
                        if showFig: (fig or plt).show()
                        if saveFig:
                            of_suff = str(self.locs.of_stem)
                            of_suff = ".".join([of_suff,str(whoami()),epochs0,ica0])
                            of_suff = ".".join([of_suff,STATUS])
                            of_suff = ".".join([of_suff,"{:03d}".format(ii)])
                            of_suff = ".".join([of_suff,"png"])
                            ## CAUTION: using a subdir here
                            of_name = od_name/of_suff
                            self.BATCH.logger.info (space0[1]+"of_name: {}".format(repr(str( of_name ))))
                            fig.savefig(of_name, dpi=fig.dpi,)




            def apply_projections_and_interpolate_bads(
                    self,
                    ica0    = "ica0",
                    epochs0 = "epochs0",
                    epochs1 = "epochs1",
                    epochs2 = "epochs2",
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"applying ICA and reference projections, and interpolating bads")
                self.BATCH.logger.info (space0[1]+"processing: {}".format(repr(str( self     ))))
                self.BATCH.logger.info (space0[1]+"ica0: {}"      .format(repr(str( ica0     ))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"   .format(repr(str( epochs0  ))))
                self.BATCH.logger.info (space0[1]+"epochs1: {}"   .format(repr(str( epochs1  ))))
                self.BATCH.logger.info (space0[1]+"epochs2: {}"   .format(repr(str( epochs2  ))))

                self.BATCH.logger.info (space0[1]+"loading data for epochs0: {}".format(repr(str(epochs0))))
                self.data[epochs0].load_data()

                self.BATCH.logger.info (space0[1]+"applying ica INC/EXC on epochs0: {}".format(repr(str(epochs0))))
                self.BATCH.logger.info (space0[2]+"saving data to epochs1: {}"         .format(repr(str(epochs1))))
                self.data[epochs1] = self.data[ica0].apply(
                    self.data[epochs0].copy(),
                )
                reset_bads = True
                mode       = "accurate"
                self.BATCH.logger.info (space0[2]+"reset_bads: {}".format(repr(str( reset_bads ))))
                self.BATCH.logger.info (space0[2]+"mode: {}"      .format(repr(str( mode       ))))

                self.BATCH.logger.info (space0[1]+"applysing projections ETC to data in epochs1: {}".format(repr(str(epochs1))))
                self.BATCH.logger.info (space0[2]+"saving data to epochs2: {}"         .format(repr(str(epochs2))))
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

                self.BATCH.logger.info (space0[1]+"DONE")




            def write_hkl(
                    self,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting dataset as hickle")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                # self.of_data = self.of_base.with_suffix(".gzip.hkl")
                # self.of_data
                # self.of_base.with_suffix(".gzip.hkl")
                hkl.dump(
                    self.data,
                    self.locs.of_data,
                    mode="w",
                    compression="gzip",
                )




            def write_data(
                    self,
                    overwrite=False,
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"exporting dataset as hickle")
                self.BATCH.logger.info (space0[1]+"processing: " + repr(str(self)))
                for key0,val0 in self.data.items():
                    self.BATCH.logger.info (space0[1]+"writing: " + repr(str(key0)))
                    self.BATCH.logger.info (space0[2]+"type: "    + repr(str(type(val0))))

                    of_suff = ""
                    if isinstance(val0, mne.io.brainvision.brainvision.RawBrainVision):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = ".".join([of_suff,"raw.fif"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        val0.save(of_name,overwrite=overwrite)

                    elif isinstance(val0, mne.epochs.BaseEpochs):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = "-".join([of_suff,"epo.fif"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        val0.save(of_name,overwrite=overwrite)

                    elif isinstance(val0, mne.preprocessing.ICA):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = "-".join([of_suff,"ica.fif"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        val0.save(of_name,overwrite=overwrite)

                    elif isinstance(val0, mne.evoked.Evoked):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = "-".join([of_suff,"ave.fif"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        val0.save(of_name,overwrite=overwrite)
                        # mne.write_evokeds

                    elif isinstance(val0, pd.DataFrame):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = "-".join([of_suff,"DataFrame.csv"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        # val0.save(of_name,overwrite=overwrite)
                        ## TODO FIXME

                    elif isinstance(val0, dict):
                        of_suff = ".".join([of_suff,str(whoami()),key0])
                        of_suff = ".".join([of_suff,"gzip.hkl"])
                        of_name = self.locs.of_base.with_suffix(of_suff)
                        hkl.dump(
                            val0,
                            of_name,
                            mode="w",
                            compression="gzip",
                        )


                    else:
                        pass
                        # raise NotImplementedError





            def evoked_to_dataframe(
                    self,
                    evoked0,     # evoked0     = "evoked0"
                    quest0,      # quest0      = "word_set"             # OR "word_len"
                    df0_evoked0, # df0_evoked0 = "df0_evoked0_word_set" # OR "df0_evoked0_word_len"
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"converting evoked to pandas dataframe")
                self.BATCH.logger.info (space0[1]+"processing: {}" .format( repr(str( self        ))))
                self.BATCH.logger.info (space0[1]+"evoked0: {}"    .format( repr(str( evoked0     ))))
                self.BATCH.logger.info (space0[1]+"quest0: {}".format( repr(str( quest0 ))))
                self.BATCH.logger.info (space0[1]+"df0_evoked0: {}".format( repr(str( df0_evoked0 ))))


                df0 = pd.DataFrame()

                for ii,(cond0_key0,cond0_val0) in enumerate(self.data[evoked0][quest0].items()):
                    assert isinstance(cond0_val0, mne.evoked.Evoked), "PROBLEM: all items in evoked0 should be an instance of mne.evoked.Evoked"
                    temp0 =  cond0_val0.to_data_frame(
                        time_format=None,
                        long_format=True,
                    )
                    # temp0["STEM"] = str(self.locs.of_stem)
                    temp0["SUB"]  = self.sub
                    temp0["SES"]  = self.ses
                    temp0["TASK"] = self.task
                    temp0["RUN"]  = self.run
                    temp0["COND"] = cond0_key0
                    self.BATCH.logger.info (space0[1]+"PROC: " + str(cond0_key0))
                    df0 = df0.append(temp0)

                di0 = self.BATCH.dataBase.setup["chans"]["bund0"]
                di1 = OrderedDict()
                for key0,val0 in di0.items():
                    for item in val0:
                        di1[item] = key0

                df0["CHAN_BUND"] = df0["channel"].apply(lambda x: di1[x] if x in di1 else None)

                self.data[df0_evoked0] = dc(df0)




            def epochs_to_dataframe(
                    self,
                    epochs0,     # epochs0     = "epochs0"
                    events0,     # events0     = "events0"
                    meta1,       # meta1       = "meta1"
                    df0_epochs0, # df0_epochs0 = "df0_epochs0"
            ):
                self.BATCH.logger.info(
                    space0[0]+"RUNNING: {}.{}".format(
                        ".".join(self.INSP),
                        str(whoami()),
                ))
                self.BATCH.logger.info (space0[1]+"converting epochs to pandas dataframe")
                self.BATCH.logger.info (space0[1]+"processing: {}" .format(repr(str(self))))
                self.BATCH.logger.info (space0[1]+"epochs0: {}"    .format(str(epochs0)))
                self.BATCH.logger.info (space0[1]+"events0: {}"    .format(str(events0)))
                self.BATCH.logger.info (space0[1]+"meta1: {}"      .format(str(meta1)))
                self.BATCH.logger.info (space0[1]+"df0_epochs0: {}".format(str(df0_epochs0)))

                assert isinstance(self.data[epochs0], mne.epochs.BaseEpochs), "PROBLEM: epochs0 hould be an instance of mne.epochs.BaseEpochs (not {})".format(str(type(epochs0)))

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

                    df0["CHAN_BUND"] = df0["channel"].apply(lambda x: di1[x] if x in di1 else None)

                self.data[df0_epochs0] = dc(df0)
