from dataloader import DataLoader
from UserCF import UserBasedCF
from ItemCF import ItemBasedCF
from LFM import LFM
from random_pred import RandomPredict
from most_popular import MostPopular
from utils import ModelManager

def get_model(model_name, movie_titles):
    if model_name == 'UserCF':
        return UserBasedCF(movie_titles=movie_titles)
    elif model_name == 'ItemCF':
        return ItemBasedCF(movie_titles=movie_titles)
    elif model_name == 'Random':
        return RandomPredict(movie_titles=movie_titles)
    elif model_name == 'MostPopular':
        return MostPopular(movie_titles=movie_titles)
    elif model_name == 'UserCF-IIF':
        return UserBasedCF(use_iif_similarity=True, movie_titles=movie_titles)
    elif model_name == 'ItemCF-IUF':
        return ItemBasedCF(use_iuf_similarity=True, movie_titles=movie_titles)
    elif model_name == 'LFM':
        return LFM(10, 20, 0.1, 0.01, 10, movie_titles=movie_titles)
    else:
        raise ValueError('Unsupported model name: ' + model_name)


def load_or_train_model(model_name, dataset_name, test_size=0.1, clean=False):
    movie_titles = DataLoader.load_movie_titles(dataset_name)
    manager = ModelManager(dataset_name, test_size)
    
    try:
        trainset = manager.load_model("trainset")
    except OSError:
        ratings = DataLoader.load_dataset(dataset_name)
        trainset, _ = DataLoader.train_test_split(ratings, test_size)
        manager.save_model(trainset, "trainset")

    if clean:
        manager.clean_workspace(True)

    try:
        model = manager.load_model(model_name)
    except OSError:
        model = get_model(model_name, movie_titles)
        model.fit(trainset)
        manager.save_model(model, model_name)

    return model, movie_titles
