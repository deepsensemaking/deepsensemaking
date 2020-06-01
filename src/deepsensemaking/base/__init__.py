#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import pandas as pd
import logging
import inspect
import pathlib

logger = logging.getLogger(__name__)

logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)

conHnd = logging.StreamHandler()
conHnd.setLevel(logging.DEBUG)
conHnd.setFormatter(
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
for handler in logger.handlers[:]:
    logger.removeHandler(handler)


# Add stream handler
logger.addHandler(conHnd)

def check_logger():
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    log = dsm.base.logger
    log.setLevel(dsm.base.logging.INFO)
    log.info("dsm version: " + str(dsm.__version__))
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
