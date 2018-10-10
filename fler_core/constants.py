# coding: utf-8

# Globals
###############################################################################

# commons

# commons on metadata dataframe
NO_LANGUAGE = 'No language specified'
NO_NAME = 'Default Name'

# commons on the features
CAP = 'Cap'
FULL_CAP = 'Full_Cap'
LENGTH = 'Length'
LOWERCASE = 'Lower_case'
PRESUFIX = 'Pre_suffixe'
# commons on the dataframe columns names
WORD = 'Word'

# commons on the prefix/suffix feature
ENG_PREFIX = ['anti', 'co', 'dis', 'il', 'im', 'in', 'inter', 'ir', 'mis', 'over', 'out', 'post', 'pre', 'pro', 'sub',
              'super', 'trans', 'under']
ENG_SUFFIX = ['dom', 'ship', 'hood', 'ian', 'er', 'er', 'ism', 'en', 'less', 'ish', 'ful', 'al', 'ly', 'en', 'ness',
              'ship', 'ity', 'ize', 'ly']

# commons on categories
LOC = 'LOC'
PER = 'PER'
MISC = 'MISC'
ORG = 'ORG'

# list features
list_features = ['Cap', 'Full_Cap', 'Length', 'Pre_suffixe',
                 'GAZMISC', 'GAZPER', 'GAZLOC', 'Number', 'FreqNAMES', 'FreqORG',
                 'FreqLOC', 'FreqPER', 'FreqMISC', 'preuniORG', 'preuniLOC', 'preuniPER',
                 'preuniMISC', 'postuniORG', 'postuniLOC', 'postuniPER', 'postuniMISC', "debut"]
