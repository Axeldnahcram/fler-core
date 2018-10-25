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
import numpy as np
import logzero
# custom
from fler_utils.commons import get_asset_root, get_file_content
import fler_core.constants as cst
from fler_core.features.features_eng import Feature_eng, features_no_directory_eng
from sklearn.externals import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import inspect

# Globals
###############################################################################

LOGGER = logzero.logger


# Functions and Classes
###############################################################################

class Training_database(object):
    """
    basic class to either, get the already trained database, or clean the dirty database
    """

    def __init__(self):
        self.df_name = None
        self.language = ""
        self.df = None
        self.categories = None
        self.entity = []
        self.trained_model = None

    @classmethod
    def get_already_trained(cls, name: str, language: str = cst.NO_LANGUAGE,
                            entity: list = ["ORG", "LOC", "PER", "MISC"]) -> Union[object, None]:
        """
        charge an already trained dataset, you must specify the langage
        :param langage: langage of the dataset
        :type langage: str
        :return: the Training database object
        :rtype: Union[object, None]
        """
        self = Training_database()
        self.df_name = name
        self.language = language
        self.entity = entity
        cfg = get_asset_root()
        file = get_file_content(cfg, name)
        try:
            self.df = pd.read_csv(filepath_or_buffer=file, index_col=0)
            self.df = self.df.fillna(0)
            self.categories = list(self.df['NEtag'].unique())
            self.categories.remove(0)
            return self
        except:
            return "Error when trying to read the dataframe"

    @classmethod
    def clean_and_setup_training(cls, name: str, language: str = cst.NO_LANGUAGE) -> Union[object, None]:
        """
        charge a virgin dataset, and add the features
        :param name: name of the virgin dataset
        :type name: str
        :param language: language of the dataset
        :type language: str
        :return: the new traning dataset object
        :rtype: Union[object, None]
        """
        self = Training_database()
        self.df_name = name
        self.language = language
        cfg = get_asset_root()
        file = get_file_content(cfg, name)
        try:
            self.df = pd.read_csv(filepath_or_buffer=file, index_col=0)
        except:
            return "Error when trying to read the dataframe"
        return self

    def do_all_train_features(self, lang: str = "ENG"):
        df = self.df
        df = Feature_eng.lowercase(df)
        df = Feature_eng.capitalize(df)
        df = Feature_eng.fullcap(df)
        df = Feature_eng.length(df)
        df = Feature_eng.presufixe(df, lang)
        df = Feature_eng.gazetteer(df, lang)
        df = Feature_eng.frequency_train(df, liste_NP=self.entity)
        df = Feature_eng.number(df)
        df = Feature_eng.preuni_train(df, liste_NP=self.entity)
        df = Feature_eng.postuni_train(df, liste_NP=self.entity)
        self.df = df
        return self.df

    def train_with_function(self, model, df: None, features: list, name_to_export: None,
                            entity: list = ['ORG', 'LOC', 'MISC', 'PER']):
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
        self.entity = entity
        if df is not None:
            self.df = df
        train_model = self.df
        f = 1
        for i in range(0, len(train_model['Word'])):
            for j in range(0, len(self.entity)):
                if train_model.loc[i, self.entity[j]] == 1:
                    multi.append(j + 1)
            if len(multi) < f:
                multi.append(0)
            f = f + 1
        LOGGER.info(f)
        LOGGER.info({len(train_model['Word']): "len trained model", len(multi): "len multi"})
        train_model['multi'] = multi
        X = train_model[features]
        y = train_model['multi']
        params_sk = model.fit(X, y)
        if name_to_export is not None:
            g = get_asset_root()
            joblib.dump(params_sk, f'{g["pkl_root"]}/{name_to_export}.pkl', compress=9)
        LOGGER.info(model.score(X, y))
        self.trained_model = params_sk
        return params_sk

    @staticmethod
    def do_feature_dataset(df: pd.DataFrame(), language: str = 'ENG', features=features_no_directory_eng):
        for i in features:
            LOGGER.info(df.head(6))
            df = i(df)
        df = Feature_eng.frequency_factory(df, "freq_names_CONLL2003")
        df = Feature_eng.preuni_factory(df, "pre_freq_CONLL2003")
        df = Feature_eng.postuni_factory(df, "post_freq_CONLL2003")
        return df

    @staticmethod
    def try_trained_model(txt_to_test: str, model: str = "svm_all_features",
                          entity: list = ['ORG', 'LOC', 'MISC', 'PER'],
                          name_entity: list = ['Organisations', 'Locations', 'Miscellaneaous', 'Persons']):
        df_to_do = pd.DataFrame()
        df_to_do['Word'] = txt_to_test.split()
        df_to_do = Training_database.do_feature_dataset(df_to_do)
        cfg = get_asset_root()
        directory = get_file_content(cfg, model)
        model_clone = joblib.load(directory)
        df_test = df_to_do[cst.list_features_en]
        result = model_clone.predict(df_test)
        LOGGER.info(result)
        dict_entity = {}
        for j in name_entity:
            dict_entity[j] = []
        for i in range(0, len(result)):
            for j in range(0, len(entity)):
                if result[i] == j + 1:
                    dict_entity[name_entity[j]].append(df_to_do['Word'][i])

        return dict_entity


if __name__ == "__main__":
    cfg = get_asset_root()
    # directory = get_file_content(cfg, "French_own_data/Reuters_fr")
    # df = pd.read_csv(directory)
    # df= df.dropna()
    # df = df.reset_index()
    # df = Training_database.do_feature_dataset(df)
    # g = Training_database()
    # rdm_forest = RandomForestClassifier(n_estimators=20, verbose=True)
    # svm_multi = svm.SVC(kernel='rbf', C=1, verbose=True)
    # h = g.train_with_function(rdm_forest, df, features=cst.list_features, name_to_export="rdm_forest_with_debut")
    # h = g.train_with_function(svm_multi, df, features=cst.list_features, name_to_export="svm_multi_easy")
    text = 'Nikky halley resigned today from the UN as her former ambassador. She might run for governor of california'
    g = Training_database.try_trained_model(text, "rdm_forest_with_debut")
    LOGGER.info(g)
