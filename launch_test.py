from tender_predictor import SimpleTenderPredictor

# Ваші дані
my_tenders = [
    {
        "edrpou": "12345678",
        "tender_name": "Закупівля запчастин",
        "item_name": "Фільтр масляний",
        "industry_name": "Автозапчастини",
        "cpv": "09211000"
    },
    # ... більше тендерів
]

# Прогнозування
predictor = SimpleTenderPredictor("tender_system.pkl")
results = predictor.predict_batch(my_tenders, threshold=0.6)

# Результати
predictor.print_results(results)

# Отримати тільки ті, що > 60%
high_probability_tenders = results['high_probability']