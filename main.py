# -*- coding = utf-8 -*-
import utils
from ItemCF import ItemBasedCF
from LFM import LFM
from UserCF import UserBasedCF
from dataloader import DataLoader
from most_popular import MostPopular
from random_pred import RandomPredict
from utils import LogTime
from dataloader import DataLoader

def run_model(model_name, dataset_name, test_size=0.3, clean=False, movie_titles=None):
    print('*' * 70)
    print('\tThis is %s model trained on %s with test_size = %.2f' % (model_name, dataset_name, test_size))
    print('*' * 70 + '\n')
    model_manager = utils.ModelManager(dataset_name, test_size)
    try:
        trainset = model_manager.load_model('trainset')
        testset = model_manager.load_model('testset')
    except OSError:
        ratings = DataLoader.load_dataset(name=dataset_name)
        trainset, testset = DataLoader.train_test_split(ratings, test_size=test_size)
        model_manager.save_model(trainset, 'trainset')
        model_manager.save_model(testset, 'testset')
    '''Do you want to clean workspace and retrain model again?'''
    '''if you want to change test_size or retrain model, please set clean_workspace True'''
    model_manager.clean_workspace(clean)
    if model_name == 'UserCF':
        model = UserBasedCF(movie_titles=movie_titles)
    elif model_name == 'ItemCF':
        model = ItemBasedCF(movie_titles=movie_titles)
    elif model_name == 'Random':
        model = RandomPredict(movie_titles=movie_titles)
    elif model_name == 'MostPopular':
        model = MostPopular(movie_titles=movie_titles)
    elif model_name == 'UserCF-IIF':
        model = UserBasedCF(use_iif_similarity=True,  movie_titles=movie_titles)
    elif model_name == 'ItemCF-IUF':
        model = ItemBasedCF(use_iuf_similarity=True,  movie_titles=movie_titles)
    elif model_name == 'LFM':
        # K, epochs, alpha, lamb, n_rec_movie
        model = LFM(10, 20, 0.1, 0.01, 10, movie_titles=movie_titles)
    else:
        raise ValueError('No model named ' + model_name)
    
    try:
        model = model_manager.load_model(model_name)
        print(f"[INFO] 成功加载模型 {model_name}.pkl")
    except OSError:
        print(f"[WARN] 未找到模型 {model_name}.pkl，正在重新训练并保存...")
    model.fit(trainset)
    model_manager.save_model(model, model_name)
    #model.fit(trainset)     # train or load a model from .pkl file
    recommend_test(model, [1, 100, 233, 666, 888])
    print("预测结果示例：")
    predictions = model.predict({str(u): None for u in [1, 100, 233, 666, 888]})
    for user, movies in predictions.items():
        print(f"user {user} 推荐：\n {movies}")
    model.test(testset)


def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        # print("recommend for userid = %s:" % user)
        # print(recommend)
        # print()


if __name__ == '__main__':
    main_time = LogTime(words="Main Function")
    dataset_name = 'ml-100k'
    movie_titles = DataLoader.load_movie_titles(dataset_name)
    # dataset_name = 'ml-1m'

    #model_name = 'UserCF'
    #model_name = 'ItemCF'
    #model_name = 'UserCF-IIF'
    #model_name = 'ItemCF-IUF'
    #model_name = 'Random'
    #model_name = 'MostPopular'
    model_name = 'LFM'    #基于矩阵分解的协同过滤算法

    test_size = 0.1

    run_model(model_name, dataset_name, test_size, False, movie_titles)

    main_time.finish()

