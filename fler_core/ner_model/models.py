# coding: utf-8
"""
.. module:: fler_core.training_df.py
basic class for a training database
"""

__author__ = "Axel Marchand"

# standard
import asyncio
from typing import Union, List, Dict
import pandas as pd
import os
import numpy as np
import logzero
from sklearn.externals import joblib
# custom
from fler_utils.commons import get_asset_root, get_file_content, get_type_of_gazetteers
import fler_core.constants as cst
from fler_core.features.features import Feature_eng
import fler_core.commons as com
import collections
import jsonpickle
import json

# Globals
################################################################################

LOGGER = logzero.logger


# Functions and Classes
###############################################################################

class NER_model(object):
    """
    class that deals with training the entity recognition model
    """
    def __init__(self, train_model: pd.DataFrame):
        self.train_model = train_model

    def train_with_function(self, model, name_entity:list, features:list, name_to_export:None):
        """
        train the model on the given named entity
        :param model: model to use, ex random forest
        :type model: sklearn stuff
        :param name_entity: list of the columns of the named entity in the dataset
        :type name_entity: list
        :param features: list of the features on which sklearn will do its stuff
        :type features: columns
        :return: trained model
        :rtype: saves the model in the pkl file
        """
        multi = []
        train_model = self.train_model
        for i in range(0, len(self.train_model['Word'])):
            f = 1
            for j in range(0, len(name_entity)):
                if self.train_model.loc[i,name_entity] == 1:
                    multi.append(j)
            if len(multi)<f:
                multi.append(0)
            f = f+1
        train_model['multi'] = multi
        X = train_model[features]
        y = train_model['multi']
        params_sk = model.fit(X,y)
        if name_to_export is not None:
            g = get_asset_root()
            joblib.dump(params_sk, f'{g["pkl_root"]}/{name_to_export}.pkl', compress=9)
        LOGGER.info(model.score(X,y))
        return params_sk




