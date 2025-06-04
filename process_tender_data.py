from process_large_files import process_large_files

# Ваші файли
files = [
    "path/to/your/first_3.5gb_file.jsonl",
    "path/to/your/second_3.5gb_file.jsonl"
]

# Запуск обробки
stats = process_large_files(
    file_paths=files,
    categories_file="categories.jsonl",
    mapping_file="category_mappings.json",
    batch_size=2000,  # Збільште якщо є RAM
    max_records_per_file=100000  # Для тесту, потім видаліть
)
