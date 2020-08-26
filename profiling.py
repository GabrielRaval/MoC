#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 02:48:05 2020

@author: gabriel
"""

import cProfile, pstats, io
from pstats import SortKey
# import MoC
with cProfile.Profile() as pr:
    import MoC

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# print(s.getvalue())
with open('profile_results.txt', 'w') as f:
    f.write(s.getvalue())
