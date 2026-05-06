# -*- coding = utf-8 -*-
"""
The :mod:`dataset` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.

"""
import collections
import os
import itertools
import random
from collections import namedtuple

BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'sep', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path='data/ml-100k/u.data',
            sep='\t',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path='data/ml-1m/ratings.dat',
            sep='::',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
}

# modify the random seed will change dataset spilt.
# if you want to use the model saved before, please don't modify this seed.
random.seed(0)


class DataLoader:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the below
    available methods for loading datasets.
    """

    def __init__(self):
        pass

    @classmethod
    def load_dataset(cls, name='ml-100k'):
        """Load a built-in dataset.

        :param name:string: The name of the built-in dataset to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.
        :return: ratings for each line.
        """
        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')
        if not os.path.isfile(dataset.path):
            raise OSError(
                "Dataset data/" + name + " could not be found in this project.\n"
                                         "Please download it from " + dataset.url +
                ' manually and unzip it to data/ directory.')
        with open(dataset.path) as f:
            ratings = [cls.parse_line(line, dataset.sep) for line in itertools.islice(f, 0, None)]
        print("Load " + name + " dataset success.")
        return ratings

    @classmethod
    def parse_line(cls, line: str, sep: str):
        """
        Parse a line.

        Ratings as ensured to positive integers.

        the separator in rating.data is `::`.

        :param sep: the separator between fields. Example : ``';'``.
        :param line: The line to parse

        :return: triple: User id, item id, rating score.
                The timestamp will be ignored since it wasn't used in Collaborative filtering.
        """
        user, movie, rate = line.strip('\r\n').split(sep)[:3]
        return user, movie, rate

    @classmethod
    def train_test_split(cls, ratings, test_size=0.2):
        """
        Split rating data to training set and test set.

        The default `test_size` is the test percentage of test size.

        The rating file should be an instance of DataSet.

        @param ratings: raw dataset
        @param test_size: the percentage of test size.
        @return: train_set and test_set
        """
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainset_len = 0
        testset_len = 0
        for user, movie, rate in ratings:
            if random.random() <= test_size:    # random simpling
                test[user][movie] = int(rate)
                testset_len += 1
            else:
                train[user][movie] = int(rate)
                trainset_len += 1
        print('split rating data to training set and test set success.')
        print('train set size = %s' % trainset_len)
        print('test set size = %s\n' % testset_len)
        return train, test
    
    @classmethod
    def load_movie_titles(cls, name='ml-100k'):
        """
        Load movie id to title mapping.

        :param name: Dataset name, either 'ml-100k' or 'ml-1m'.
        :return: dict {movie_id: title}
        """
        movie_titles = {}

        if name == 'ml-100k':
            path = 'data/ml-100k/u.item'
            sep = '|'
            id_index = 0
            title_index = 1
        elif name == 'ml-1m':
            path = 'data/ml-1m/movies.dat'
            sep = '::'
            id_index = 0
            title_index = 1
        else:
            raise ValueError(f"Unsupported dataset name: {name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Movie metadata file not found: {path}")

        with open(path, encoding='ISO-8859-1') as f:
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) > title_index:   
                    try:
                        movie_id = int(parts[id_index])  # 👈 强制转换为 int 类型
                        title = parts[title_index]
                        movie_titles[movie_id] = title
                    except ValueError:
                        continue  # 跳过无法转换为 int 的行# movie_id = parts[id_index]
                    # title = parts[title_index]
                    # movie_titles[movie_id] = title

        print(f"Loaded movie titles from {name}: {len(movie_titles)} entries.")
        return movie_titles

