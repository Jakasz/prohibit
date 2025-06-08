import hashlib

filename = "out_10.jsonl"  # замініть на ваш шлях до файлу
output_file = "out_10_nodup.jsonl"
seen_hashes = {}
duplicates = []

# Підрахунок дублікатів
with open(filename, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line_hash = hashlib.md5(line.strip().encode("utf-8")).hexdigest()
        if line_hash in seen_hashes:
            seen_hashes[line_hash] += 1
            duplicates.append(line_num)
        else:
            seen_hashes[line_hash] = 1

print(f"Знайдено {len(duplicates)} повних дублікатів.")
if duplicates:
    print("Рядки з дублікатами:", duplicates)

# Видалення дублікатів (залишає тільки перший запис)
with open(filename, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    written_hashes = set()
    for line in fin:
        line_hash = hashlib.md5(line.strip().encode("utf-8")).hexdigest()
        if line_hash not in written_hashes:
            fout.write(line)
            written_hashes.add(line_hash)

print(f"Файл без дублікатів збережено як {output_file}")
print("Кількість унікальних записів:", len(written_hashes))