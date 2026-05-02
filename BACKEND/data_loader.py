import csv

class DataLoader:
    def __init__(self):
        self.users = self.load_users()
        self.products = self.load_products()
        self.ratings = self.load_ratings()
        self.behavior = self.load_behavior()

    def load_users(self):
        data = []
        with open("../users_cleaned.xlsx", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "user_id": int(row["user_id"]),
                    "age": int(row["age"]),
                    "country": row["country"]
                })
        return data

    def load_products(self):
        data = []
        with open("../products_cleaned.xlsx", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "product_id": int(row["product_id"]),
                    "category": row["category"],
                    "price": float(row["price"])
                })
        return data

    def load_ratings(self):
        data = []
        with open("../ratings_cleaned.xlsx", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "user_id": int(row["user_id"]),
                    "product_id": int(row["product_id"]),
                    "rating": int(row["rating"])
                })
        return data

    def load_behavior(self):
        data = []
        with open("../behavior_cleaned.xlsx", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "user_id": int(row["user_id"]),
                    "product_id": int(row["product_id"]),
                    "clicked": int(row["clicked"]),
                    "viewed": int(row["viewed"]),
                    "purchased": int(row["purchased"]),
                    "click_throu": int(row["click_throu"]),
                    "conversion_rate": int(row["conversion_rate"])
                })
        return data
