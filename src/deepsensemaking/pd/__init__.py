#!/usr/bin/env python
# -*- coding: utf-8 -*





"""
Example usage of multiple context managers

  import pandas as pd
  from IPython.display    import display,HTML,Image
  from deepsensemaking.pd import mgrs
  from contextlib         import ExitStack

  with ExitStack() as stack:
      [stack.enter_context(mgr) for mgr in mgrs]
      print( pd.get_option("display.max_rows") )
      display(df0)

"""


from contextlib import ExitStack
import pandas as pd
mgrs = [
    pd.option_context("display.max_columns"  ,   45),
    pd.option_context("display.max_colwidth" ,   80),
    pd.option_context("display.width"        ,  800),
    pd.option_context("display.max_rows"     ,  200),
    pd.option_context("display.min_rows"     ,  200),
]

