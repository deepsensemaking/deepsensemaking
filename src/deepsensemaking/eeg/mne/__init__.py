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

import warnings
import logging


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


from   deepsensemaking.bids    import get_bids_prop
from   deepsensemaking.dicts   import str_dict,print_dict

import mne
mne.set_log_level("WARNING")



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

    log0.info("got: "+str(value))
    return value


class BatchMNE:
    """MNE batch job class"""
    def __init__(
            self,
            batchName   = "DS",
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
        assert isinstance(batchName,(str,)),"expected batchName to be of type \"string\", got {}".format(str(type(batchName)))
        assert isinstance(verbose,(int,float,complex,)),"expected verbose to be a number, got {}".format(str(type(verbose)))
        assert sourceDir.   exists()  ,"provided sourceDir"   + "path does not exist"
        assert setupFile.   exists()  ,"provided setupFile"   + "path does not exist"
        assert stimuliFile. exists()  ,"provided stimuliFile" + "path does not exist"
        assert sourceDir.   is_dir()  ,"provided sourceDir"   + "path is not a directory"
        assert setupFile.   is_file() ,"provided setupFile"   + "path is not a file"
        assert stimuliFile. is_file() ,"provided stimuliFile" + "path is not a file"

        ## Basic class attributes
        self.batchName   = batchName
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
        self.logger = logging.getLogger(__name__)
        """
        self.logger.setLevel(logging.CRITICAL) # 50
        self.logger.setLevel(logging.ERROR)    # 40
        self.logger.setLevel(logging.WARNING)  # 30
        self.logger.setLevel(logging.INFO)     # 20
        self.logger.setLevel(logging.DEBUG)    # 10
        self.logger.setLevel(logging.NOTSET)   # 00
        """
        self.logger.setLevel(logging.DEBUG)    # 10
        handler0 = logging.StreamHandler()
        handler0.setLevel(logging.INFO)
        handler0.setFormatter(
            logging.Formatter(": ".join([
                # "%(asctime)s",
                # "%(name)s",
                "%(levelname)s",
                "%(message)s",
            ]),
            datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        fn0 = self.loggerDir/(dt.datetime.now(loc_tz).strftime("%Y%m%d_%H%M%S_%f")[:-3]+".log")
        handler1 = logging.FileHandler(fn0)
        handler1.setLevel(logging.DEBUG)
        handler1.setFormatter(
            logging.Formatter(": ".join([
                "%(asctime)s",
                "%(name)s",
                "%(levelname)s",
                "%(message)s",
            ]),
            datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        self.logger.addHandler(handler0)
        self.logger.addHandler(handler1)
        self.logger.info ("\n"+" "*2+"logging to: "+str(fn0))
        self.logger.debug("\n"+" "*2+"handler0 level: "+str(logging.getLevelName(handler0)))
        self.logger.debug("\n"+" "*2+"handler1 level: "+str(logging.getLevelName(handler1)))
        self.logger.info ("\n"+" "*2+"MNE version: " + str(mne.__version__))

        self.inputPaths  = self.InputPaths(self)
        self.dataBase    = self.DataBase(self)

    def info(self):
        self.logger.info(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out_str  = ""
        out_str += "\n"+" "*2+self.batchName+".batchName   = "+str(self.batchName  )
        out_str += "\n"+" "*2+self.batchName+".sourceDir   = "+str(self.sourceDir  )
        out_str += "\n"+" "*2+self.batchName+".targetDir   = "+str(self.targetDir  )
        out_str += "\n"+" "*2+self.batchName+".loggerDir   = "+str(self.loggerDir  )
        out_str += "\n"+" "*2+self.batchName+".globSuffix  = "+str(self.globSuffix )
        out_str += "\n"+" "*2+self.batchName+".globPattern = "+str(self.globPattern)
        out_str += "\n"+" "*2+self.batchName+".setupFile   = "+str(self.setupFile  )
        out_str += "\n"+" "*2+self.batchName+".stimuliFile = "+str(self.stimuliFile)
        out_str += "\n"+" "*2+self.batchName+".verbose     = "+str(self.verbose    )
        out_str += "\n"+" "*2+self.batchName+".logger      : "+str(self.logger     )
        out_str += "\n"+" "*2+self.batchName+".inputPaths  : "+"contains {} items".format(len(self.inputPaths))
        out_str += "\n"+" "*2+self.batchName+".dataBase    : "+"contains {} items".format(len(self.dataBase))
        return out_str

    def __len__(self):
        return self.inputPaths.__len__()

    class InputPaths(UserList):
        def __init__(self,BATCH):
            UserList.__init__(self)
            self.BATCH = BATCH

        def glob(self):
            self.BATCH.logger.info("\n"+" "*2+"* Getting paths")
            self.BATCH.logger.info("\n"+" "*2+"** Using pattern: "+str(self.BATCH.globPattern))
            self.data = glob.glob(str(self.BATCH.globPattern))

        def select(self,selector,mode="keep"):
            selector = pathlib.Path(selector)
            assert selector.exists(), "Provided selector file not found!"
            self.BATCH.logger.info("\n"+" "*2+"* Selecting input paths")
            self.BATCH.logger.info("\n"+" "*2+"** File: {}".format(selector))
            self.BATCH.logger.info("\n"+" "*2+"** Mode: {}".format(mode))
            selecting_strings = list()
            with open(selector) as fh0:
                for line in fh0:
                    selecting_strings.append(line.strip())

            if selecting_strings:
                if mode=="keep":
                    self.data = [ item for item in self.data if     any( item for selector in selecting_strings if selector in item ) ]
                else:
                    self.data = [ item for item in self.data if not any( item for selector in selecting_strings if selector in item ) ]

        def numel(self):
            self.BATCH.logger.info("\n"+" "*2+"Input paths count: {}".format(len(self.data)))

        def info_str(self):
            out_str  = ""
            if self.data:
                for item in self.data:
                    out_str += "\n"+" "*2+item
            else:
                out_str = "\n"+" "*2+"No input paths to display!"

            return out_str

        def info(self):
            self.BATCH.logger.info(self.info_str())

    class DataBase(UserList):
        def __init__(self,BATCH):
            UserList.__init__(self)
            self.BATCH = BATCH
            self.setup = OrderedDict()

        def get_setup(self):
            fn0 = self.BATCH.setupFile
            with open(fn0) as fh0:
                self.setup = OrderedDict(json.load(fh0))

        def setup_info(self):
            self.BATCH.logger.info(str_dict(self.setup," "*2+self.BATCH.batchName+".setup"))

        def get_stimuli(self):
            fn1 = self.BATCH.stimuliFile
            self.stimuli = pd.read_csv(fn1)

        def stimuli_info(self):
            self.BATCH.logger.info(self.stimuli)

        def get_paths(self):
            self.BATCH.logger.info("\n"+" "*2+"* Getting data paths")
            self.data = list()
            for idx,item in enumerate(self.BATCH.inputPaths):
                self.data.append(
                    self.DataSet(
                        BATCH = self.BATCH,
                        item  = item,
                    )
                )

        def read_ALL(self,key0="raw0",preload=True,verbose=None):
            for idx,item in enumerate(self.data):
                self.data[idx].read_raw_data(key0=key0,preload=preload,verbose=verbose)

        def info_str(self):
            out_str  = ""
            out_str += self.BATCH.batchName+".data: "+str(len(self.data))+"\n"
            for idx,item in enumerate(self.data):
                temp_status = "[    ]" if item.locs.of_done.is_file() else "[TODO]"
                #out_str += " "*2+str(item.locs.of_stem)+": "+str(len(item.data.keys()))+"\n"
                out_str += "  {:>{}d}: {} {}".format(idx,len(str(len(self.data)-1)),temp_status,item,)
                out_str += "\n"
                out_str += "  {}         {}: {}".format(" "*len(str(len(self.data)-1)),len(item),repr(list(item.data.keys())))
                out_str += "\n"

            return out_str

        def info(self):
             self.BATCH.logger.info(self.info_str())




        class DataSet():
            """
            DataSet.data["raw0"]
            DataSet.locs.if_path
            DataSet.locs.of_path
            DataSet.locs.od_path
            DataSet.locs.of_stem
            DataSet.locs.of_base
            DataSet.locs.of_bads
            DataSet.locs.of_done
            """
            def __init__(self,BATCH,item):
                self.BATCH = BATCH
                self.locs  = self.Locs(BATCH,item)
                self.data  = OrderedDict()
                os.makedirs(self.locs.od_path,mode=0o700,exist_ok=True,)

            def __repr__(self):
                return self.__str__()

            def __str__(self):
                return str(self.locs.of_stem)

            def __len__(self):
                return len(self.data.keys())

            def get_keys(self):
                return list(self.data.keys())

            def info_str(self):
                out_str  = ""
                out_str += "\n"+" "*2+self.__str__()
                out_str += "\n"+" "*4+str(self.get_keys())
                return out_str

            def info(self):
                self.BATCH.logger.info(self.info_str())

            class Locs:
                def __init__(self,BATCH,item):
                    self.BATCH   = BATCH
                    self.if_path = pathlib.Path(item)
                    self.of_path = BATCH.targetDir / self.if_path.relative_to(BATCH.sourceDir)
                    self.od_path = self.of_path.parents[0]
                    self.of_stem = pathlib.Path(self.of_path.stem.split('.')[0])
                    self.of_base = self.od_path / self.of_stem
                    self.of_bads = self.of_base.with_suffix(".bads")
                    self.of_done = self.of_base.with_suffix(".gzip.hkl")

                def __repr__(self):
                    return self.__str__()

                def __str__(self):
                    out_str  = ""
                    out_str += "\n"+" "*2+self.BATCH.batchName+".data[IDX].locs"
                    out_str += "\n"+" "*4+"if_path: "+str(self.if_path)
                    out_str += "\n"+" "*4+"of_done: "+str(self.of_done)
                    out_str += "\n"+" "*4+"of_base: "+str(self.of_base)
                    out_str += "\n"+" "*4+"of_bads: "+str(self.of_bads)
                    out_str += "\n"+" "*4+"of_done: "+str(self.of_done)
                    return out_str

                def info(self):
                    BATCH.logger.info(self.__str__())


        ## =============================================================================
        ## Utilities
        ## =============================================================================


        def read_raw_data(self,raw0="raw0",preload=True,verbose=None):
            self.BATCH.logger.info("\n"+" "*2+"* Reading raw data")
            self.BATCH.logger.info("\n"+" "*2+"** Processing: " + str(self))
            data_input = self.locs.if_path
            self.data["proclog"] = OrderedDict()
            self.data["proclog"]["data_input"] = str(data_input)
            self.data[raw0] = mne.io.read_raw_brainvision(
                vhdr_fname = data_input,
                eog        = ['HEOGL','HEOGR','VEOGb'],
                misc       = 'auto',
                scale      = 1.0,
                preload    = preload,
                verbose    = verbose,
            )
            self.data[raw0].info["description"] = str(self.locs.of_stem)

        def check_chans_number(self,raw0="raw0",):
            chan_num_expected = self.data["META"]["chans"]["init"]
            log0.info("Checking channel numbers consistency")
            log0.info(" "*2+"Processing: " + str(self))
            temp_chans = len(self.data[raw0].copy().pick_types(meg=False,eeg=True).ch_names)
            assert chan_num_expected == temp_chans, " ".join([
                "Problem occured",
                "while reading '{}'".format(str(self.data[raw0])),
                "data was expected to contain {} EEG channels,".format(chan_num_expected,),
                "but {} were found!".format(temp_chans,),
            ])
            self.data["proclog"]["check_chans_number"] = True
            return True

        def check_BAD_chans_file(self,raw0="raw0",):
            bad_names = list()
            of_bads   = self.locs.of_bads
            log0.info("Checking BAD channels information file")
            log0.info(" "*2+"Processing: " + str(self))
            log0.debug(" "*4+"Looking for: " + str(of_bads))
            if os.path.exists(of_bads):
                log0.info(" "*4+"Found bad channels file...")
                self.data["proclog"]["bad_chans_file"] = of_bads
                with open(of_bads) as fh:
                    for line in fh:
                        line = line.split('#',1,)[0].strip()
                        if line:
                            bad_names.append(line)

            if bad_names:
                self.data[raw0].info['bads'] += bad_names
                self.data[raw0].info['bads'] = list(set(self.data[raw0].info['bads']))
                log0.info(" "*4+"Added bad channels informtion to raw data")

            self.data["proclog"]["bad_chans_list"] = bad_names
            return True

        def average_reference_projection(self,raw0="raw0",montage="standard_1005",ref_channels = "average",):
            log0.info("Adding Actual Reference Channel(s)")
            log0.info(" "*2+"Processing: " + str(self))
            log0.info("Adding {} to DATA".format(self.data["META"]["chans"]["refs"]))
            log0.info(" "*2+"Processing: " + str(self))
            mne.add_reference_channels(
                inst         = self.data[raw0],
                ref_channels = self.data["META"]["chans"]["refs"],
                copy         = False,
            )
            self.data["proclog"]["reference_channels"] = self.data["META"]["chans"]["refs"]
            log0.info("Setting Montage")
            log0.info("Using {}".format(montage))
            log0.info(" "*2+"Processing: " + str(self))
            self.data[raw0].set_montage(montage=montage,)
            log0.info("Setting Average Reference Projection")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[raw0].set_eeg_reference(
                ref_channels = ref_channels,
                projection   = True,
                ch_type      = "eeg",
            )
            self.data["proclog"]["set_eeg_reference"] = ref_channels


        def process_events_and_annotations(self,raw0="raw0",annots0="annots0",events0="events0"):
            log0.info("Processing events and annotations")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[annots0]         = OrderedDict()
            self.data[annots0]["orig"] = self.data[raw0].annotations.copy()
            self.data[annots0]["orig"].save(str(self.locs.of_base.with_suffix(".raw0.annots0.orig.csv")))
            (temp_event_time,
             temp_event_desc) = mne.events_from_annotations(self.data[raw0],)
            self.data[events0]         = OrderedDict()
            self.data[events0]["event_desc"] = temp_event_desc
            self.data[events0]["event_time"] = temp_event_time
            self.data["proclog"]["process_annots"] = True
            self.data["proclog"]["process_events"] = True


        def check_for_BAD_spans(self,raw0="raw0",annots1="annots1"):
            log0.info("Checking for Bad Spans")
            log0.info(" "*2+"Processing: " + str(self))
            of_annot1 = str(self.locs.of_base.with_suffix(".raw0.annots1.bad_spans.csv"))
            if os.path.exists(of_annot1):
                log0.info(" "*4+"Found bad span annottions file!")
                self.data[annots1]              = OrderedDict()
                self.data[annots1]["bad_spans"] = mne.read_annotations(of_annot1)
                self.data[raw0].set_annotations(self.data[raw0].annotations + self.data[annots1]["bad_spans"])
                log0.info("    Added bad span annottions to raw data")
                self.data["proclog"]["annots1_updated"] = True
                self.data["proclog"]["annots1_from"] = of_annot1
            else:
                self.data["proclog"]["annots1_updated"] = False


        def bandpass_filter(self,raw0="raw0"):
            log0.info("Applying Bandpass Filter to Data")
            log0.info(" "*2+"Processing: " + str(self))
            time_t0 = time.time()
            self.data[raw0].filter(
                l_freq     = self.data["META"]["filt"]["l_freq"],
                h_freq     = self.data["META"]["filt"]["h_freq"],
                fir_design = self.data["META"]["filt"]["fir_design"],
            )
            time_t1 = time.time()
            time_d1 = time_t1-time_t0
            log0.info(" "*4+"Time Elapsed: " + hf.format_timespan( time_d1 ))
            self.data["proclog"]["filter"] = True
            self.data["proclog"]["l_freq"] = self.data["META"]["filt"]["l_freq"]
            self.data["proclog"]["h_freq"] = self.data["META"]["filt"]["h_freq"]
            self.data["proclog"]["fir_design"] = self.data["META"]["filt"]["fir_design"]


        def plot_channels_power_spectral_density(self,raw0="raw0",average=False,exclude=True):
            log0.info("Plotting channel power spectral density")
            log0.info(" "*2+"Processing: " + str(self))
            # plt.close('all')
            of_suff = ""
            of_suff = ".".join([of_suff,raw0,"plot_psd"])
            of_suff = ".".join([of_suff,"aggrChn"]) if average else ".".join([of_suff,"eachChn"])
            of_suff = ".".join([of_suff,"exclBAD"]) if exclude else ".".join([of_suff,"inclBAD"])
            of_suff = ".".join([of_suff,"png"])
            EXCLUDE = self.data[raw0].info["bads"]  if exclude else []
            picks   = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=EXCLUDE,)
            fig = self.data[raw0].plot_psd(
                show    = False,
                fmin    =  0,
                fmax    = 60,
                picks   = picks,
                average = average,
                proj    = True,
            )
            fig.set_size_inches(16,8)
            title_orig = fig.axes[0].get_title()
            title_pref = str(self.locs.of_stem)
            fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
            # plt.tight_layout(pad=.5)
            plt.show()
            fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
            # plt.close()


        def plot_raw_data_timeseries(self,raw0="raw0",total=False,exclude=True):
            log0.info("Plotting raw data timeseries")
            log0.info(" "*2+"Processing: " + str(self))
            # plt.close('all')
            of_suff = ""
            of_suff = ".".join([of_suff,raw0,"plot_raw"])
            of_suff = ".".join([of_suff,"fullSig"]) if total   else ".".join([of_suff,"someSig"])
            of_suff = ".".join([of_suff,"exclBAD"]) if exclude else ".".join([of_suff,"inclBAD"])
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
            title_orig = fig.axes[0].get_title()
            title_pref = str(self.locs.of_stem)
            fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
            # plt.tight_layout(pad=.5)
            plt.show()
            if total:
                fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
                # plt.close()

        def export_BAD_spans_info(self,raw0="raw0"):
            log0.info("Exporting BAD spans annotation data to CSV file")
            log0.info(" "*2+"Processing: " + str(self))
            bads_annot1 = [0]+[ii for ii,an in enumerate(self.data[raw0].annotations) if an['description'].lower().startswith("bad")]
            of_bads_annot1 = str(self.locs.of_base.with_suffix(".raw0.annots1.bad_spans.csv"))
            of_bads_annot2 = str(self.locs.of_base.with_suffix(".raw0.annots2.new_check.csv"))
            self.data[raw0].annotations[bads_annot1].save(of_bads_annot1)
            self.data[raw0].annotations[          :].save(of_bads_annot2)
            self.data["proclog"]["of_bads_annot1"] = of_bads_annot1
            self.data["proclog"]["of_bads_annot2"] = of_bads_annot2


        def extract_metadata_for_acquired_events(self,events0="events0"):
            log0.info("Extracting Metadata for Acquired Events")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[events0]["event_meta"] = pd.DataFrame(self.data[events0]["event_time"], columns=["ONSET","DURATION","CODE"] )
            # self.data[events0]["event_meta"] = self.data[events0]["event_meta"][ (self.data[events0]["event_meta"]["CODE"] < 300) &  (self.data[events0]["event_meta"]["CODE"] > 100) ]
            self.data[events0]["event_meta"]["DIFF"] = self.data[events0]["event_meta"]["ONSET"].diff()
            self.data[events0]["event_meta"]["DIFF"] = self.data[events0]["event_meta"]["DIFF"].fillna(0)
            self.data[events0]["event_meta"] = pd.merge(
                left      = self.data[events0]["event_meta"],
                right     = self.data["META"]["events"]["dgn"]["0"],
                how       = "left",
                left_on   = "CODE",
                right_on  = "CODE",
                sort      = False,
                suffixes  = ("_acq","_dgn"),
                copy      = True,
                indicator = False,
                validate  = "m:1",
            )
            self.data[events0]["event_meta"]["FILE"] = str(self.locs.of_stem)
            self.data[events0]["event_meta"]["SUB"]  = get_bids_prop(if_name=str(self.locs.of_stem),prop="sub",)
            self.data[events0]["event_meta"]["RUN"]  = get_bids_prop(if_name=str(self.locs.of_stem),prop="run",)
            self.data[events0]["event_meta"]["TASK"] = get_bids_prop(if_name=str(self.locs.of_stem),prop="task",)
            ## Select only stimulus related events
            self.data[events0]["EVENT_META"] = self.data[events0]["event_meta"][ (self.data[events0]["event_meta"]["CODE"] < 300) &  (self.data[events0]["event_meta"]["CODE"] > 100) ]
            self.data[events0]["EVENT_TIME"] = self.data[events0]["EVENT_META"][["ONSET","DURATION","CODE"]].to_numpy()
            self.data[events0]["EVENT_DESC"] = self.data[events0]["event_desc"]
            try: del self.data[events0]["EVENT_DESC"]['Comment/no USB Connection to actiCAP']
            except: pass
            try: del self.data[events0]["EVENT_DESC"]['New Segment/']
            except: pass
            try: del self.data[events0]["EVENT_DESC"]['Stimulus/S  1']
            except: pass
            try: del self.data[events0]["EVENT_DESC"]['Stimulus/S  2']
            except: pass
            try: del self.data[events0]["EVENT_DESC"]['Stimulus/S  3']
            except: pass
            self.data["proclog"]["extract_metadata_for_acquired_events"] = True
            self.data["proclog"]["events0_meta_acquired"] = True


        def construct_epochs(self,raw0="raw0",events0="events0",epochs0="epochs0"):
            log0.info("Constructing epochs")
            log0.info(" "*2+"Processing: " + str(self))
            exclude = self.data[raw0].info["bads"] + self.data["META"]["chans"]["refs"]
            exclude = self.data["META"]["chans"]["refs"]
            exclude = self.data[raw0].info["bads"]
            exclude = []
            picks   = mne.pick_types(self.data[raw0].info,meg=False,eeg=True,exclude=exclude,)
            self.data[epochs0] = mne.Epochs(
                raw      = self.data[raw0].copy(),
                events   = self.data[events0]["EVENT_TIME"],
                event_id = self.data[events0]["EVENT_DESC"],
                metadata = self.data[events0]["EVENT_META"],
                tmin     = -0.200,
                tmax     =  0.900,
                baseline = (None, 0),
                picks    = picks,
                # reject   = reject,
                preload  = True,
                reject_by_annotation=True,
                reject   = self.data["META"]["params"]["reject"],
                flat     = self.data["META"]["params"]["flat"],
                decim    = 5,
            )
            self.data["proclog"]["construct_epochs"] = True


        def inspect_epochs(self,epochs0="epochs0"):
            log0.info("Inspect epochs (MARK BAD)")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[epochs0].plot()
            self.data["proclog"]["epochs_inspected"] = True


        def plot_epochs_drop_log(self,epochs0="epochs0"):
            log0.info("Inspect epochs (MARK BAD)")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
            fig = self.data[epochs0].plot_drop_log(show=False)
            fig.set_size_inches(8,4)
            title_orig = fig.axes[0].get_title()
            title_pref = str(self.locs.of_stem)
            fig.axes[0].set(title='\n'.join([title_pref, title_orig]))
            plt.show()
            of_suff  = ""
            of_suff += "."+epochs0
            of_suff += ".plot_drop_log0.0.png"
            fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
            # plt.close()
            self.data["proclog"]["plot_epochs_drop_log"] = True


        def plot_epochs_AGGREGATED(self,epochs0="epochs0"):
            log0.info("Plot epochs AGGREGATED")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
            combines = ["gfp","mean",]
            for jj,combine in enumerate(combines):
                # .copy().apply_proj()
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
                    fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
                    # plt.close()


        def plot_epochs_BUNDLES(self,epochs0="epochs0"):
            log0.info("Plot epochs BUNDLES")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
            elec_idx = OrderedDict()
            for key,val in self.data["META"]["chans"]["bund"]["0"].items():
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
                    fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
                    # plt.close()


        def construct_evoked(self,evoked0="evoked0",epochs0="epochs0"):
            log0.info("Construct evoked")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[evoked0]            = OrderedDict()
            self.data[evoked0]["default"] = OrderedDict()
            self.data[evoked0]["default"]["all_in"] = self.data[epochs0].average()
            for ii,(key,val) in enumerate(self.data["META"]["queries"]["default"].items()):
                self.data[evoked0]["default"][key]   = self.data[epochs0][val].average()


        def plot_evoked(self,evoked0="evoked0"):
            log0.info("Plot evoked")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
            spatial_colors = False
            spatial_colors = True
            times  = "auto"
            times  = list()
            times += [-0.200,-0.100,0,]
            times += self.data["META"]["time"]["means"]["0"].values()
            # times += list(sum(self.data["META"]["time"]["spans"]["0"].values(), ()))
            times = sorted(times)
            for ii,(key,evoked) in enumerate(self.data[evoked0]["default"].items()):
                title = ""
                title += "EEG ({})".format( evoked.info["nchan"] - len(evoked.info["bads"]) )
                title += str(self.locs.of_stem)
                fig = evoked.copy(
                ).apply_proj(
                ).interpolate_bads(
                    reset_bads=True,mode="accurate",
                ).plot_joint(
                    title   = title + "\n{}".format(key),
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
                fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
                # plt.close()


        def run_ica(self,ica0="ica0",epochs0="epochs0"):
            log0.info("Run ICA")
            log0.info(" "*2+"Processing: " + str(self))
            time_T0 = time.time()
            self.data[ica0] = mne.preprocessing.ica.ICA(
                n_components       = 50,
                n_pca_components   = 50,
                max_pca_components = 50,
                method             = 'fastica',
                max_iter           = 500,
                # noise_cov          = noise_cov,
                random_state       = 0,
            ).fit(
                self.data[epochs0],
            )
            time_T1 = time.time()
            time_D1 = time_T1-time_T0
            log0.info(" "*4+"TIME Elapsed: " + hf.format_timespan( time_D1 ))


        def inspect_components(self,ica0="ica0",epochs0="epochs0"):
            log0.info("Inspect components")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
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
                fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)


        def plot_components(self,ica0="ica0",epochs0="epochs0",rejected=True,save=False):
            log0.info("Plot rejected components")
            log0.info(" "*2+"Processing: " + str(self))
            plt.close('all')
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
                        of_suff  = ".".join([of_suff,"exc"]) if rejected else ".".join([of_suff,"inc"])
                        of_suff += ".0-{:02d}.png".format(ii)
                        fig.savefig(self.locs.of_base.with_suffix(of_suff), dpi=fig.dpi,)
                        # plt.close()


        def apply_projections_and_interpolate_bads(self,ica0="ica0",epochs0="epochs0",epochs1="epochs1",epochs2="epochs2"):
            log0.info("Apply ICA, projections and interpolate bads")
            log0.info(" "*2+"Processing: " + str(self))
            self.data[epochs1] = self.data[ica0].apply(self.data[epochs0].copy())
            reset_bads = True
            mode       = "accurate"
            self.data[epochs2] = self.data[epochs1].copy(
            ).apply_proj(
            # ).resample(
            #     sfreq=200,
            ).interpolate_bads(
                reset_bads = reset_bads,
                mode       = mode,
            )
            log0.debug("="*77)
            log0.debug(self.data[epochs0].info)
            log0.debug("="*77)
            log0.debug(self.data[epochs1].info)
            log0.debug("="*77)
            log0.debug(self.data[epochs2].info)
