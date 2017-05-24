from pipeline_controller import *
from feature_controller import *
from pipeline_builder import *
from visualization_toolbox import *
from data_holder import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob


class model_controller(object):
    """ GridSearchCV extension with model comparision
    """
    def __init__(self):
        return 1

    def param_edit_pop():
        self.param_grid.pop()

    def param_edit_add(additional_params):
        self.param_grid.append(additional_params)

    def model_selection_statistics(self):
        """ In case of TimeSeries Data

        ARIMA
        """
        return None
