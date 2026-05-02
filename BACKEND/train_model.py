"""
Run this ONCE locally to train and save the model.
This creates model.pkl which will be uploaded to Render.
"""
import pickle
import os

from ga_engine import ExcelDatabase, DataPreprocessor, GeneticRecommender

current_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading databases...")
db_users = ExcelDatabase(os.path.join(current_dir, 'users_cleaned.xlsx'))
db_products = ExcelDatabase(os.path.join(current_dir, 'products_cleaned.xlsx'))
db_ratings = ExcelDatabase(os.path.join(current_dir, 'ratings_cleaned.xlsx'))
db_behavior = ExcelDatabase(os.path.join(current_dir, 'behavior_cleaned.xlsx'))

if not all([db_users.connect(), db_products.connect(), 
            db_ratings.connect(), db_behavior.connect()]):
    print("Failed to connect to databases!")
    exit()

print("Building preprocessor...")
preprocessor = DataPreprocessor(db_users, db_products, db_ratings, db_behavior)

print("Training Genetic Algorithm...")
ga = GeneticRecommender(preprocessor, population_size=20, elite_count=2, tournament_size=3)
best = ga.run(generations=5)

print(f"Training complete! Best fitness: {best.fitness:.4f}")

# Save ONLY the best chromosome (not the whole GA)
model_data = {
    'best_chromosome': best,
    'preprocessor_note': 'Loaded separately from Excel files'
}

with open(os.path.join(current_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved to model.pkl")
print("Upload this file to GitHub along with your code!")
