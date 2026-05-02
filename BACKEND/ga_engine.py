import pandas as pd
import os
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

# ============================================================================
# PART 1: EXCEL DATABASE CLASS
# ============================================================================

class ExcelDatabase:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
    
    def connect(self):
        """Load Excel data"""
        try:
            self.df = pd.read_excel(self.filepath)
            print(f"✓ Connected to {os.path.basename(self.filepath)}: {len(self.df)} rows")
            print(f"  Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{self.filepath}' not found")
            print(f"  Current directory: {os.getcwd()}")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def get_all_ids(self, id_column):
        """Get all unique IDs from a column"""
        if self.df is not None and id_column in self.df.columns:
            return self.df[id_column].unique().tolist()
        return []
    
    def get_by_id(self, id_column, id_value):
        """Get row(s) by ID"""
        if self.df is not None:
            return self.df[self.df[id_column] == id_value]
        return pd.DataFrame()


# ============================================================================
# PART 2: CHROMOSOME CLASS
# ============================================================================

class Chromosome:
    def __init__(self, genes=None):
        if genes:
            self.genes = genes.copy()
        else:
            self.genes = {
                'w_rating': random.uniform(0.0, 3.0),
                'w_category': random.uniform(0.0, 3.0),
                'w_price': random.uniform(-2.0, 2.0),
                'w_age': random.uniform(0.0, 2.0),
                'w_country': random.uniform(0.0, 2.0),
                'w_click': random.uniform(0.0, 4.0),
                'w_view': random.uniform(0.0, 2.0),
                'w_purchase': random.uniform(0.0, 8.0),
                'k_neighbors': random.randint(5, 50),
                'alpha': random.uniform(0.0, 1.0),
                'recency_decay': random.uniform(0.5, 1.0)
            }
        self.fitness = None
        self.id = random.randint(10000, 99999)
    
    def copy(self):
        """Create a copy of this chromosome"""
        return Chromosome(self.genes)
    
    def __repr__(self):
        fit_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Chromo({self.id}, α={self.genes['alpha']:.2f}, fit={fit_str})"


# ============================================================================
# PART 3: DATA PREPROCESSOR (Optimized with Precomputation)
# ============================================================================

class DataPreprocessor:
    def __init__(self, db_users, db_products, db_ratings, db_behavior):
        self.db_users = db_users
        self.db_products = db_products
        self.db_ratings = db_ratings
        self.db_behavior = db_behavior
        
        # Fast lookup dictionaries
        self.user_profiles = {}
        self.product_info = {}
        self.user_ratings = defaultdict(dict)
        self.product_ratings = defaultdict(dict)
        self.user_behaviors = defaultdict(lambda: {'clicked': set(), 'viewed': set(), 'purchased': set()})
        
        # Precomputed structures for speed
        self.user_similarity_matrix = {}
        self.user_normalized = {}
        self.product_normalized = {}
        self.user_rated_products = {}
        self.user_category_prefs = defaultdict(lambda: defaultdict(int))
        self.avg_price = 500
        self.max_price = 1000
        
        self.build_lookups()
        self.precompute_similarities()
        self.precompute_normalized_features()
    
    def build_lookups(self):
        """Build fast lookup dictionaries from dataframes"""
        print("\n--- Building Fast Lookups ---")
        
        # Users lookup
        if self.db_users.df is not None:
            for _, row in self.db_users.df.iterrows():
                self.user_profiles[row['user_id']] = {
                    'age': row['age'],
                    'country': row['country']
                }
            print(f"✓ Loaded {len(self.user_profiles)} users")
        
        # Products lookup
        if self.db_products.df is not None:
            # Check column name
            cat_col = 'category' if 'category' in self.db_products.df.columns else 'catagory'
            for _, row in self.db_products.df.iterrows():
                self.product_info[row['product_id']] = {
                    'category': row[cat_col],
                    'price': row['price']
                }
            print(f"✓ Loaded {len(self.product_info)} products")
        
        # Ratings lookup
        if self.db_ratings.df is not None:
            for _, row in self.db_ratings.df.iterrows():
                self.user_ratings[row['user_id']][row['product_id']] = row['rating']
                self.product_ratings[row['product_id']][row['user_id']] = row['rating']
            print(f"✓ Loaded {len(self.db_ratings.df)} ratings")
        
        # Behavior lookup
        if self.db_behavior.df is not None:
            for _, row in self.db_behavior.df.iterrows():
                uid, pid = row['user_id'], row['product_id']
                if row['clicked'] == 1:
                    self.user_behaviors[uid]['clicked'].add(pid)
                if row['viewed'] == 1:
                    self.user_behaviors[uid]['viewed'].add(pid)
                if row['purchased'] == 1:
                    self.user_behaviors[uid]['purchased'].add(pid)
                    # Track category preferences
                    if pid in self.product_info:
                        cat = self.product_info[pid]['category']
                        self.user_category_prefs[uid][cat] += 1
            print(f"✓ Loaded behaviors for {len(self.user_behaviors)} users")
    
    def precompute_similarities(self):
        """Precompute user-user similarities"""
        print("--- Precomputing user similarities ---")
        user_ids = list(self.user_ratings.keys())
        
        for i, u1 in enumerate(user_ids):
            u1_rated = set(self.user_ratings[u1].keys())
            similarities = []
            for u2 in user_ids:
                if u1 == u2:
                    continue
                u2_rated = set(self.user_ratings[u2].keys())
                intersection = u1_rated & u2_rated
                if len(intersection) >= 3:
                    union = u1_rated | u2_rated
                    sim = len(intersection) / len(union) if union else 0
                    similarities.append((u2, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.user_similarity_matrix[u1] = similarities
            
            if (i+1) % 200 == 0:
                print(f"  Processed {i+1}/{len(user_ids)} users...")
        print(f"✓ Precomputed similarities for {len(self.user_similarity_matrix)} users")
    
    def precompute_normalized_features(self):
        """Precompute normalized features"""
        # Products
        categories = list(set(info['category'] for info in self.product_info.values()))
        cat_to_code = {cat: i for i, cat in enumerate(categories)}
        
        prices = [info['price'] for info in self.product_info.values()]
        self.avg_price = sum(prices) / len(prices) if prices else 500
        self.max_price = max(prices) if prices else 1000
        
        for pid, info in self.product_info.items():
            self.product_normalized[pid] = {
                'category_code': cat_to_code[info['category']],
                'price_norm': (info['price'] - self.avg_price) / max(self.max_price, 1)
            }
        
        # Users
        countries = list(set(prof['country'] for prof in self.user_profiles.values()))
        country_to_code = {c: i for i, c in enumerate(countries)}
        
        for uid, prof in self.user_profiles.items():
            self.user_normalized[uid] = {
                'age_norm': (prof['age'] - 35) / 50,
                'country_code': country_to_code[prof['country']]
            }
        
        # Pre-store rated products
        for uid in self.user_ratings:
            self.user_rated_products[uid] = list(self.user_ratings[uid].keys())
        
        print(f"✓ Normalized features for {len(self.product_normalized)} products and {len(self.user_normalized)} users")
    
    def get_top_k_similar_users(self, user_id, k):
        """Get top K similar users (precomputed)"""
        if user_id not in self.user_similarity_matrix:
            return []
        return self.user_similarity_matrix[user_id][:k]
    
    def get_user_category_match(self, user_id, category):
        """Get user's preference for a category"""
        if user_id not in self.user_category_prefs:
            return 0.5
        prefs = self.user_category_prefs[user_id]
        total = sum(prefs.values())
        return prefs.get(category, 0) / total if total > 0 else 0.5


# ============================================================================
# PART 4: GENETIC RECOMMENDER
# ============================================================================

class GeneticRecommender:
    def __init__(self, preprocessor, population_size=30, elite_count=2, tournament_size=3):
        self.preprocessor = preprocessor
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
    
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            self.population.append(Chromosome())
        print(f"✓ Initialized population of {self.population_size} chromosomes")
    
    def calculate_bh_score(self, user_id, product_id, chromosome):
        """Calculate Behavioral score using Behavior DB (Fast)"""
        genes = chromosome.genes
        behaviors = self.preprocessor.user_behaviors.get(user_id, {
            'clicked': set(), 'viewed': set(), 'purchased': set()
        })
        
        score = 0
        if product_id in behaviors['clicked']:
            score += genes['w_click']
        if product_id in behaviors['viewed']:
            score += genes['w_view']
        if product_id in behaviors['purchased']:
            score += genes['w_purchase']
        return score
    
    def calculate_cf_score_fast(self, user_id, product_id, chromosome):
        """Optimized CF score using precomputed similarities"""
        genes = chromosome.genes
        similar_users = self.preprocessor.get_top_k_similar_users(user_id, genes['k_neighbors'])
        
        if not similar_users:
            return 2.5 * genes['w_rating']
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        product_ratings = self.preprocessor.product_ratings.get(product_id, {})
        for sim_user, sim in similar_users:
            rating = product_ratings.get(sim_user)
            if rating is not None:
                weighted_sum += sim * rating
                weight_total += sim
        
        if weight_total > 0:
            return (weighted_sum / weight_total) * genes['w_rating']
        return 2.5 * genes['w_rating']
    
    def calculate_cb_score_fast(self, user_id, product_id, chromosome):
        """Optimized CB score using precomputed normalized features"""
        genes = chromosome.genes
        user_norm = self.preprocessor.user_normalized.get(user_id)
        prod_norm = self.preprocessor.product_normalized.get(product_id)
        
        if not user_norm or not prod_norm:
            return 0.0
        
        score = 0.0
        
        # Category match
        product_cat = self.preprocessor.product_info[product_id]['category']
        user_cat_pref = self.preprocessor.get_user_category_match(user_id, product_cat)
        score += genes['w_category'] * user_cat_pref
        
        # Price sensitivity
        price_factor = -prod_norm['price_norm']
        score += genes['w_price'] * price_factor
        
        # Age factor
        score += genes['w_age'] * (1.0 - abs(user_norm['age_norm']))
        
        # Country factor
        score += genes['w_country'] * 0.5
        
        return score
    
    def predict_score_fast(self, user_id, product_id, chromosome):
        """Fast prediction using precomputed data"""
        genes = chromosome.genes
        cf = self.calculate_cf_score_fast(user_id, product_id, chromosome)
        cb = self.calculate_cb_score_fast(user_id, product_id, chromosome)
        bh = self.calculate_bh_score(user_id, product_id, chromosome)
        return genes['alpha'] * cf + (1 - genes['alpha']) * (cb + bh)
    
    def evaluate_fitness_optimized(self, chromosome, test_users):
        """Optimized fitness evaluation"""
        total_reward = 0.0
        total_predictions = 0
        
        for user_id in test_users:
            purchased = self.preprocessor.user_behaviors[user_id]['purchased']
            if not purchased:
                continue
            
            # Candidate products
            candidates = set(self.preprocessor.user_rated_products.get(user_id, []))
            if len(candidates) < 30:
                all_prods = list(self.preprocessor.product_info.keys())
                needed = 30 - len(candidates)
                candidates.update(random.sample(all_prods, min(needed, len(all_prods))))
            
            # Score candidates
            scores = {}
            for pid in candidates:
                if pid not in purchased:
                    scores[pid] = self.predict_score_fast(user_id, pid, chromosome)
            
            if not scores:
                continue
            
            # Top 10
            top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            user_purchases = self.preprocessor.user_behaviors[user_id]['purchased']
            user_clicks = self.preprocessor.user_behaviors[user_id]['clicked']
            
            for rec_prod, _ in top_10:
                if rec_prod in user_purchases:
                    total_reward += 10
                elif rec_prod in user_clicks:
                    total_reward += 2
                total_predictions += 1
        
        return total_reward / len(test_users) if total_predictions > 0 else 0.0
    
    def tournament_select(self):
        """Select one parent using tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda c: c.fitness if c.fitness is not None else 0)
    
    def clip_genes(self, genes):
        """Ensure genes stay within valid ranges"""
        ranges = {
            'w_rating': (0.0, 3.0),
            'w_category': (0.0, 3.0),
            'w_price': (-2.0, 2.0),
            'w_age': (0.0, 2.0),
            'w_country': (0.0, 2.0),
            'w_click': (0.0, 4.0),
            'w_view': (0.0, 2.0),
            'w_purchase': (0.0, 8.0),
            'k_neighbors': (5, 50),
            'alpha': (0.0, 1.0),
            'recency_decay': (0.5, 1.0)
        }
        
        for key in genes:
            if key in ranges:
                min_val, max_val = ranges[key]
                genes[key] = max(min_val, min(max_val, genes[key]))
                if key == 'k_neighbors':
                    genes[key] = int(genes[key])
    
    def crossover(self, parent1, parent2):
        """Uniform crossover with alpha blending"""
        child_genes = {}
        
        for key in parent1.genes:
            if isinstance(parent1.genes[key], float):
                alpha = 0.5
                min_val = min(parent1.genes[key], parent2.genes[key])
                max_val = max(parent1.genes[key], parent2.genes[key])
                range_val = max_val - min_val
                child_genes[key] = random.uniform(
                    min_val - alpha * range_val,
                    max_val + alpha * range_val
                )
            else:
                child_genes[key] = parent1.genes[key] if random.random() < 0.5 else parent2.genes[key]
        
        self.clip_genes(child_genes)
        return Chromosome(child_genes)
    
    def mutate(self, chromosome, mutation_rate=0.15):
        """Apply Gaussian mutation to genes"""
        for key in chromosome.genes:
            if random.random() < mutation_rate:
                if key == 'k_neighbors':
                    chromosome.genes[key] += random.randint(-5, 5)
                else:
                    chromosome.genes[key] += random.gauss(0, 0.2)
        self.clip_genes(chromosome.genes)
    
    def evolve_generation(self):
        """Run one generation of evolution"""
        print(f"\n--- Generation {self.generation} ---")
        
        test_users = random.sample(list(self.preprocessor.user_profiles.keys()), 
                                   min(20, len(self.preprocessor.user_profiles)))
        
        # Evaluate unevaluated chromosomes
        unevaluated = [c for c in self.population if c.fitness is None]
        for i, chromo in enumerate(unevaluated):
            chromo.fitness = self.evaluate_fitness_optimized(chromo, test_users)
            if (i+1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(unevaluated)}...")
        
        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness if c.fitness is not None else 0, reverse=True)
        
        best_fitness = self.population[0].fitness
        avg_fitness = sum(c.fitness for c in self.population if c.fitness is not None) / len(self.population)
        self.best_fitness_history.append(best_fitness)
        
        print(f"  Best Fitness: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")
        print(f"  Best Chromosome: {self.population[0]}")
        
        # Elitism
        new_population = [self.population[0].copy()]
        if self.elite_count > 1:
            new_population.extend([c.copy() for c in self.population[1:self.elite_count]])
        
        # Create children
        while len(new_population) < self.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        return best_fitness
    
    def run(self, generations=10):
        """Run genetic algorithm"""
        if not self.population:
            self.initialize_population()
        
        print("\n" + "="*60)
        print("STARTING GENETIC ALGORITHM EVOLUTION")
        print("="*60)
        
        for gen in range(generations):
            self.evolve_generation()
        
        print("\n" + "="*60)
        print("EVOLUTION COMPLETE")
        print("="*60)
        
        return self.population[0]
    
    def get_recommendations(self, user_id, n=10):
        """Get top N recommendations for a user"""
        best_chromosome = self.population[0]
        all_products = list(self.preprocessor.product_info.keys())
        purchased = self.preprocessor.user_behaviors[user_id]['purchased']
        
        scores = {}
        for product_id in all_products:
            if product_id not in purchased:
                scores[product_id] = self.predict_score_fast(user_id, product_id, best_chromosome)
        
        top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("GENETIC ALGORITHM RECOMMENDATION SYSTEM")
    print("="*60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n--- Loading Excel Databases ---")
    
    db_users = ExcelDatabase(os.path.join(current_dir, 'users_cleaned.xlsx'))
    db_products = ExcelDatabase(os.path.join(current_dir, 'products_cleaned.xlsx'))
    db_ratings = ExcelDatabase(os.path.join(current_dir, 'ratings_cleaned.xlsx'))
    db_behavior = ExcelDatabase(os.path.join(current_dir, 'behavior_cleaned.xlsx'))
    
    if not all([db_users.connect(), db_products.connect(), 
                db_ratings.connect(), db_behavior.connect()]):
        print("\n✗ Failed to connect to all databases. Exiting.")
        return
    
    preprocessor = DataPreprocessor(db_users, db_products, db_ratings, db_behavior)
    
    ga = GeneticRecommender(
        preprocessor=preprocessor,
        population_size=20,  # Smaller for speed
        elite_count=2,
        tournament_size=3
    )
    
    best_chromosome = ga.run(generations=5)  # Fewer generations for testing
    
    print("\n" + "="*60)
    print("BEST CHROMOSOME FOUND")
    print("="*60)
    for key, value in best_chromosome.genes.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*60)
    
    sample_users = list(preprocessor.user_profiles.keys())[:3]
    for user_id in sample_users:
        print(f"\nUser {user_id}:")
        recommendations = ga.get_recommendations(user_id, n=5)
        for product_id, score in recommendations:
            product_info = preprocessor.product_info.get(product_id, {})
            print(f"  Product {product_id}: {product_info.get('category', 'N/A')} - Score: {score:.4f}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()