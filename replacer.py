import json

def process_file(filename):
    field_null_count = {}
    total_lines = 0
    output_filename = filename.replace('.jsonl', '_nonull.jsonl')

    with open(filename, 'r', encoding='utf-8') as fin, open(output_filename, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_lines += 1
            record = json.loads(line)
            changed = False
            for key, value in record.items():
                if value is None:
                    record[key] = ""
                    field_null_count[key] = field_null_count.get(key, 0) + 1
                    changed = True
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\nФайл: {filename}")
    for field, count in field_null_count.items():
        print(f"Поле '{field}': замінено null у {count} рядках")
    if not field_null_count:
        print("null значень не знайдено")
    print(f"Всього рядків: {total_lines}")
    print(f"Результат збережено у: {output_filename}")

process_file('out_10_nodup.jsonl')
process_file('out_12_nodup.jsonl')