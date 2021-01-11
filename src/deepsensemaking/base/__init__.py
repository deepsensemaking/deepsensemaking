#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import pandas as pd
import logging
import inspect
import pathlib


import time
import datetime as dt
from pytz import timezone as tz
loc_tz = tz("Europe/Berlin")
import humanfriendly as hf
import tempfile

"""

time_t0 = time.time()
time.sleep(0.2)
time_t1 = time.time()
time_d1 = time_t1-time_t0

# print("Time Elapsed: {}".format(hf.format_timespan( time_d1 )))


# print()





"""

def dt_now():
    """
    Example usage:
    ==============
      from deepsensemaking.base import dt_now
      print("{}".format(dt_now()))

    """
    return str(dt.datetime.now(loc_tz).strftime("%Y%m%d-%H%M%S-%f")[:-3])



logger = logging.getLogger(__name__)

logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

logHnd0 = logging.StreamHandler()
logHnd0.setLevel(logging.INFO)
logHnd0.setFormatter(
    logging.Formatter(
        ": ".join([
            # "%(asctime)s",
            # "%(name)s",
            "%(levelname)s",
            "%(message)s",
        ]),
        datefmt="[%Y-%m-%d %H:%M:%S]",
    ))

# Remove all handlers associated to a logger
# NB: This alleviates problems that result from
# ipython autoreload fnctionality
# see: https://stackoverflow.com/questions/41443336/python-2-7-remove-handler-object-or-logger
# [:] important not to mutate the list during iteration over it
for handler in logger.handlers[:]:
    logger.removeHandler(handler)


# Add stream handler
logger.addHandler(logHnd0)

def check_logger():
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    log0 = dsm.base.logger
    log0.setLevel(dsm.base.logging.INFO)
    log0.info("dsm version: " + str(dsm.__version__))
    dsm.base.check_logger()
    """
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


def set_cwd(var_name="EMACS_BUFFER_DIR"):
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    _DIR = dsm.set_cwd()
    """
    dir0_path = str(pathlib.Path().resolve())
    vars_dict = inspect.stack()[1][0].f_locals
    vars_list = list(vars_dict.keys())
    logger.debug("got var names: " + str(vars_list))
    if var_name in vars_list:
        logger.info("found '{}' variable".format(var_name))
        dir1_path = str(pathlib.Path(vars_dict[var_name]).resolve())
        dir2_path = vars_dict[var_name]
        if dir0_path != dir1_path:
            logger.info("changing CWD to '{}'".format(dir2_path))
            os.chdir(vars_dict[var_name])
        else:
            logger.info("keeping '{}' as CWD".format(os.getcwd()))
    else:
        logger.info(f"no {var_name} variable was found")
        logger.info("keeping '{}' as CWD".format(os.getcwd()))
    return os.getcwd()


def outside_emacs(var_name="EMACS_BUFFER_DIR"):
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    import pandas as pd
    pd.set_option("display.notebook_repr_html", outside_emacs() )
    # To be used with org-mode
    #+BEGIN_SRC ipython :session *iPython* :eval yes :results raw drawer :exports both :shebang "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n" :var EMACS_BUFFER_DIR=(file-name-directory buffer-file-name) :tangle yes
    #+END_SRC
    """
    vars_dict = inspect.stack()[1][0].f_locals
    vars_list = list(vars_dict.keys())
    logger.debug("got var names: " + str(vars_list))
    if var_name in vars_list:
        logger.debug("outside_emacs = False")
        return False
    else:
        logger.debug("outside_emacs = True")
        return True


def cont_pd(
        max_columns  =  45,
        max_colwidth =  80,
        width        = 800,
        max_rows     =  45,
        min_rows     =  45,
):
    """
    Example use:
      import pandas as pd
      from contextlib import ExitStack
      import deepsensemaking as dsm

      with ExitStack() as stack:
        [stack.enter_context(cont) for cont in dsm.base.cont_pd()]
        print( pd.get_option("display.max_rows") )
        # display(df2)

"""

    return [
        pd.option_context("display.max_columns"  , max_columns  ),
        pd.option_context("display.max_colwidth" , max_colwidth ),
        pd.option_context("display.width"        , width        ),
        pd.option_context("display.max_rows"     , max_rows     ),
        pd.option_context("display.min_rows"     , min_rows     ),
    ]



