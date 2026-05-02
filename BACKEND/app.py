from flask import Flask, render_template, request, jsonify
import os
import sys

# Add BACKEND folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_engine import ExcelDatabase, Chromosome, DataPreprocessor, GeneticRecommender
current_dir = os.path.dirname(os.path.abspath(__file__))
# Initialize Flask - template folder is in BACKEND
app = Flask(__name__, 
            template_folder=os.path.join(current_dir, 'templates'),
            static_folder=os.path.join(current_dir, 'static'),
            static_url_path='/static')

# Global variables (initialized later)
preprocessor = None
ga = None

def initialize_system():
    """Load data and train the model - called on first request"""
    global preprocessor, ga
    
    if preprocessor is not None and ga is not None:
        return  # Already initialized
    
    print("INITIALIZING SYSTEM...")
            
    print(f"Working directory: {current_dir}")
    print(f"Files found: {os.listdir(current_dir)}")
    
    # Load databases
    db_users = ExcelDatabase(os.path.join(current_dir, 'users_cleaned.xlsx'))
    db_products = ExcelDatabase(os.path.join(current_dir, 'products_cleaned.xlsx'))
    db_ratings = ExcelDatabase(os.path.join(current_dir, 'ratings_cleaned.xlsx'))
    db_behavior = ExcelDatabase(os.path.join(current_dir, 'behavior_cleaned.xlsx'))
    
    if not all([db_users.connect(), db_products.connect(), 
                db_ratings.connect(), db_behavior.connect()]):
        raise Exception("Failed to connect to databases")
    
    # Build preprocessor
    preprocessor = DataPreprocessor(db_users, db_products, db_ratings, db_behavior)
    
    # Initialize GA with smaller values for Render free tier
    ga = GeneticRecommender(
        preprocessor=preprocessor,
        population_size=5,   # Reduced for Render
        elite_count=1,
        tournament_size=2
    )

    # Train with fewer generations
    best = ga.run(generations=2)
    print(f"✓ System initialized. Best fitness: {best.fitness:.4f}")
    return best


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


@app.route("/how_it_works.html")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/cart.html")
def cart_page():
    return render_template("cart.html")


@app.route("/api/dashboard")
def dashboard_api():
    global preprocessor
    
    if preprocessor is None:
        initialize_system()
    
    users_count = len(preprocessor.user_profiles)
    products_count = len(preprocessor.product_info)
    ratings_count = sum(len(ratings) for ratings in preprocessor.user_ratings.values())
    
    category_counts = {}
    for p in preprocessor.product_info.values():
        cat = p.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    top_category = max(category_counts, key=category_counts.get) if category_counts else "N/A"
    category_labels = list(category_counts.keys())
    category_values = list(category_counts.values())
    
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


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    global preprocessor, ga
    
    if preprocessor is None or ga is None:
        initialize_system()
    
    data = request.get_json()
    
    try:
        user_id = int(data.get("user_id"))
    except:
        return jsonify({"error": "رقم المستخدم غير صالح"}), 400
    
    if user_id not in preprocessor.user_profiles:
        return jsonify({
            "user_id": user_id,
            "recommendations": [],
            "message": "المستخدم غير موجود"
        })
    
    recs = ga.get_recommendations(user_id, n=10)
    results = []
    
    for product_id, score in recs:
        info = preprocessor.product_info.get(product_id, {})
        results.append({
            "product_id": int(product_id),
            "category": info.get("category", "N/A"),
            "price": float(info.get("price", 0)),
            "score": round(float(score), 4)
        })
    
    return jsonify({
        "user_id": user_id,
        "recommendations": results
    })


# Health check endpoint for Render
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "initialized": preprocessor is not None})


if __name__ == "__main__":
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
