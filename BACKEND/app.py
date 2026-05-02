from flask import Flask, render_template, request, jsonify
import os

from ga_engine import ExcelDatabase, DataPreprocessor, GeneticRecommender

app = Flask(__name__, template_folder="../", static_folder="../CSS")
# تحميل قواعد البيانات
current_dir = os.path.dirname(os.path.abspath(__file__))

db_users = ExcelDatabase(os.path.join(current_dir, 'users_cleaned.xlsx'))
db_products = ExcelDatabase(os.path.join(current_dir, 'products_cleaned.xlsx'))
db_ratings = ExcelDatabase(os.path.join(current_dir, 'ratings_cleaned.xlsx'))
db_behavior = ExcelDatabase(os.path.join(current_dir, 'behavior_cleaned.xlsx'))

db_users.connect()
db_products.connect()
db_ratings.connect()
db_behavior.connect()

# تجهيز البيانات والخوارزمية الجينية
preprocessor = DataPreprocessor(db_users, db_products, db_ratings, db_behavior)
ga = GeneticRecommender(preprocessor, population_size=20, elite_count=2, tournament_size=3)
ga.initialize_population()
ga.run(generations=3)


# دالة التوصيات المعتمدة على GA
def ga_recommendation(user_id, n=5):
    if user_id not in preprocessor.user_profiles:
        return []

    recs = ga.get_recommendations(user_id, n=10)
    results = []

    for product_id, score in recs:
        info = preprocessor.product_info.get(product_id, {})
        results.append({
            "product_id": int(product_id),
            "category": info.get("category", "N/A"),
            "price": float(info.get("price", 0)),
            "score": float(score)
        })

    return results


# API: Dashboard
@app.route("/api/dashboard")
def dashboard_api():
    # عدد المستخدمين
    users_count = len(preprocessor.user_profiles)

    # عدد المنتجات
    products_count = len(preprocessor.product_info)

    # عدد التقييمات
    ratings_count = sum(len(ratings) for ratings in preprocessor.user_ratings.values())

    # حساب أكثر فئة انتشاراً
    category_counts = {}
    for p in preprocessor.product_info.values():
        cat = p.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    top_category = max(category_counts, key=category_counts.get)

    # تجهيز بيانات الرسم البياني للفئات
    category_labels = list(category_counts.keys())
    category_values = list(category_counts.values())

    # تجهيز بيانات الرسم البياني للتقييمات
    rating_labels = ["1", "2", "3", "4", "5"]
    rating_values = [0, 0, 0, 0, 0]

    for user_id, ratings in preprocessor.user_ratings.items():
        for product_id, score in ratings.items():
            if 1 <= score <= 5:
                rating_values[score - 1] += 1

    return jsonify({
        "users": users_count,
        "products": products_count,
        "ratings": ratings_count,
        "top_category": top_category,
        "category_labels": category_labels,
        "category_counts": category_values,
        "rating_labels": rating_labels,
        "rating_counts": rating_values
    })


# الصفحات
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def index_page():
    return render_template("index.html")

@app.route("/about.html")
def about_page():
    return render_template("about.html")

@app.route("/contact.html")
def contact_page():
    return render_template("contact.html")

@app.route("/recommend.html")
def recommend_page():
    return render_template("recommend.html")

@app.route("/dashboard.html")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/how_it_works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/cart.html")
def cart_page():
    return render_template("cart.html")


# API: التوصيات
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()
    try:
        user_id = int(data.get("user_id"))
    except:
        return jsonify({"error": "رقم المستخدم غير صالح"}), 400

    recommendations = ga_recommendation(user_id,n=10)

    if not recommendations:
        return jsonify({
            "user_id": user_id,
            "recommendations": [],
            "message": "لا يوجد توصيات لهذا المستخدم."
        })

    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations
    })


# تشغيل السيرفر
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=5000, debug=True)