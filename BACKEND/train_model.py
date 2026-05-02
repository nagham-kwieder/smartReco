"""
Run this ONCE locally to train and save the model.
"""
import pickle
import os
import random

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

# Check how many users have purchases
users_with_purchases = sum(1 for uid in preprocessor.user_behaviors 
                           if preprocessor.user_behaviors[uid]['purchased'])
print(f"Users with purchase history: {users_with_purchases}")

if users_with_purchases == 0:
    print("ERROR: No users have purchased items. Cannot train.")
    exit()

print("Training Genetic Algorithm...")
ga = GeneticRecommender(preprocessor, population_size=10, elite_count=2, tournament_size=3)

# Run training
best = ga.run(generations=3)

# Fix: Ensure at least one chromosome is evaluated
if best is None or best.fitness is None:
    print("First attempt failed. Trying manual evaluation...")
    # Manually evaluate the first chromosome
    test_users = random.sample(
        [u for u in preprocessor.user_behaviors 
         if preprocessor.user_behaviors[u]['purchased']], 
        min(20, users_with_purchases)
    )
    if ga.population:
        ga.population[0].fitness = ga.evaluate_fitness_optimized(ga.population[0], test_users)
        best = ga.population[0]
        print(f"Manual evaluation: fitness = {best.fitness}")

if best is None or best.fitness is None:
    print("ERROR: Training failed. Cannot create model.")
    exit()

print(f"Training complete! Best fitness: {best.fitness:.4f}")
print(f"Best chromosome genes:")
for key, value in best.genes.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Save the model
model_data = {
    'best_chromosome': best,
    'fitness': best.fitness,
    'genes': best.genes
}

model_path = os.path.join(current_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to {model_path}")
print(f"  File size: {os.path.getsize(model_path)} bytes")
print("\nUpload model.pkl to GitHub along with your other files!")
