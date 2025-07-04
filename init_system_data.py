import subprocess

def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    result = subprocess.run(["python", cmd], check=True)
    print(f"✅ {desc} завершено\n")

if __name__ == "__main__":
    run_step("vector_db/create_vector_db.py", "Створення векторної бази")
    run_step("vector_db/enable_indexing.py", "Увімкнення індексації")
    run_step("build_cache_n_profiles.py", "Створення кешу та профілів")
    run_step("create_profiles_with_clusters.py", "Оновлення профілів з кластерами та конкурентами")
    print("🎉 Всі етапи завершено!")