import numpy as np
import pandas as pd
import copy

from konlpy.tag import Kkma

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

import seaborn as sns