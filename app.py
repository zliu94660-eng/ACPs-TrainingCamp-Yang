from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from LFM import LFM
from UserCF import UserBasedCF
from ItemCF import ItemBasedCF
from random_pred import RandomPredict
from utils import LogTime
from flask import Flask, request, jsonify, render_template
from model_runner import load_or_train_model

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model"

class ModelManager:
    @staticmethod
    def load_model(filename):
        path = os.path.join(MODEL_DIR, filename + ".pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        with open(path, 'rb') as f:
            return pickle.load(f)



def load_model_components(model_name, dataset_name, test_size):
    prefix = f"{dataset_name}-testsize{test_size}"
    use_iif = False

    if model_name.endswith("-IIF"):
        use_iif = True
        model_name = model_name.replace("-IIF", "")

    movie_titles = LogTime(dataset_name)

    if model_name == "UserCF":
        model = UserBasedCF(movie_titles=movie_titles, use_iif_similarity=use_iif, save_model=False)
        model.user_sim_mat = ModelManager.load_model(f"{prefix}-user_sim_mat-iif" if use_iif else f"{prefix}-user_sim_mat")
        model.movie_popular = ModelManager.load_model(f"{prefix}-movie_popular")
        model.movie_count = ModelManager.load_model(f"{prefix}-movie_count")
        model.trainset = ModelManager.load_model(f"{prefix}-trainset")
        return model

    elif model_name == "ItemCF":
        model = ItemBasedCF(movie_titles=movie_titles, use_iif_similarity=use_iif, save_model=False)
        model.movie_sim_mat = ModelManager.load_model(f"{prefix}-movie_sim_mat-iif" if use_iif else f"{prefix}-movie_sim_mat")
        model.movie_popular = ModelManager.load_model(f"{prefix}-movie_popular")
        model.movie_count = ModelManager.load_model(f"{prefix}-movie_count")
        model.trainset = ModelManager.load_model(f"{prefix}-trainset")
        return model

    elif model_name == "LFM":
        model = LFM(movie_titles=movie_titles)
        model.P = ModelManager.load_model(f"{prefix}-K=10-epochs=20-alpha=0.1-lamb=0.01-P")
        model.Q = ModelManager.load_model(f"{prefix}-K=10-epochs=20-alpha=0.1-lamb=0.01-Q")
        model.trainset = ModelManager.load_model(f"{prefix}-trainset")
        return model

    elif model_name == "RandomPredict":
        model = RandomPredict(movie_titles=movie_titles)
        model.trainset = ModelManager.load_model(f"{prefix}-trainset")
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def generate_douban_url(title):
    from urllib.parse import quote
    return f"https://www.douban.com/search?q={quote(title)}&cat=1002"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = int(data["user_id"])
    model_name = data["model_name"]
    dataset_name = data["dataset_name"]
    test_size = 0.1  # 固定 test_size

    try:
        model, movie_titles = load_or_train_model(model_name, dataset_name, test_size=test_size)
        recommendations = model.recommend(str(user_id))

        results = []
        for mid,score in recommendations:
            title = movie_titles.get(int(mid), f"Movie {mid}")
            results.append({
                "title": title,
                "score": round(score, 2),
                "detail_url": generate_douban_url(title)
            })

        if not results:
            return jsonify({"recommendations": [], "message": "无推荐结果或用户不在训练集中。"})

        return jsonify({"recommendations": results})
    except Exception as e:
        return jsonify({"error": str(e)})





@app.route("/")
def home():
    # 渲染 templates/index.html 文件
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
