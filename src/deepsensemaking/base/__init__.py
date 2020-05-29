#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import pandas as pd
import logging
import inspect

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


def set_cwd():
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    DIR_PROJ_MAIN = dsm.set_cwd()
    """
    stck = inspect.stack()[1][0]
    vard = stck.f_locals
    keys = list(vard.keys())
    key  = "DIR_PROJ_MAIN"
    logger.debug("variables: " + str(keys))
    if key in keys:
        logger.info("changing cwd: " + str(vard[key]))
        os.chdir(vard[key])
    else:
        logger.info("keeping cwd: " + str(os.getcwd()))
    return os.getcwd()


def inside_emacs():
    """
    Example usage:
    ==============
    import deepsensemaking as dsm
    import pandas as pd
    pd.set_option("display.notebook_repr_html", not dsm.inside_emacs() )
    """
    stck = inspect.stack()[1][0]
    vard = stck.f_locals
    keys = list(vard.keys())
    key  = "DIR_PROJ_MAIN"
    if key in keys:
        logger.debug("inside_emacs = True")
        return True
    else:
        logger.debug("inside_emacs = False")
        return False
