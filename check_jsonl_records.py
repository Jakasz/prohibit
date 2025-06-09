import json
import re

# Файл для перевірки
JSONL_FILE = "out_10_nodup.jsonl"
N = 1_000_000

# Аналогічні перевірки як у _prepare_point (без ембедингів)
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[\w\s\-\.\(\)]', lambda m: m.group(0), text)  # залишаємо тільки дозволені символи
    text = re.sub(r'[^\w\s\-\.\(\)]', ' ', text)  # замінюємо всі інші на пробіл
    units_mapping = {
        r'\bшт\.?\b': 'штук',
        r'\bкг\.?\b': 'кілограм',
        r'\bг\.?\b': 'грам',
        r'\bл\.?\b': 'літр',
        r'\bм\.?\b': 'метр',
        r'\bсм\.?\b': 'сантиметр',
        r'\bмм\.?\b': 'міліметр'
    }
    for pattern, replacement in units_mapping.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_record(item):
    # 1. Основні поля
    tender_number = item.get('F_TENDERNUMBER', '')
    edrpou = item.get('EDRPOU', '')
    item_name = item.get('F_ITEMNAME', '')
    if not tender_number:
        return "Пустий F_TENDERNUMBER"
    if not edrpou:
        return f"Пустий EDRPOU для тендера {tender_number}"
    if not item_name:
        return f"Пустий F_ITEMNAME для тендера {tender_number}"
    # 2. Текст для ембедингу
    tender_name = item.get('F_TENDERNAME', '')
    detail_name = item.get('F_DETAILNAME', '')
    combined_text = f"{item_name} {tender_name} {detail_name}".strip()
    if len(combined_text) < 3:
        return f"Надто короткий текст ({len(combined_text)}) для тендера {tender_number}"
    processed_text = preprocess_text(combined_text)
    if not processed_text or len(processed_text) < 2:
        return f"Після препроцесингу текст занадто короткий для тендера {tender_number}"
    # Якщо всі перевірки пройдено
    return None

def main():
    skipped = 0
    reasons = {}
    total = 0
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            if total >= N:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                skipped += 1
                reasons.setdefault("JSON decode error", 0)
                reasons["JSON decode error"] += 1
                continue
            reason = check_record(item)
            if reason:
                skipped += 1
                reasons.setdefault(reason, 0)
                reasons[reason] += 1
            total += 1
    print(f"Перевірено {total} записів. Пропущено: {skipped}")
    print("Причини пропуску:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
