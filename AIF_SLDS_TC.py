#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:23:42 2024

Switching linear dynamical system (SLDS) using an Active Inference controller for
Trajectory Control (EC)

@author: pzc
"""

from pymdp import utils
from agent import Agent
import ssm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# In[] Construct linear dynamics

