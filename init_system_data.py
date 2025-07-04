import logging
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from vector_db.create_vector_db import create_optimized_vector_database
    from vector_db.enable_indexing import enable_indexing_with_timeout
    from profiles.build_cache_n_profiles import run_build_cache
    # Assuming create_profiles_with_clusters.py has a main callable function.
    # If the refactor to run_profile_creation_with_clusters_flow was successful, use it.
    # Otherwise, fallback to the original structure if possible, adapting the call.
    try:
        from create_profiles_with_clusters import run_profile_creation_with_clusters_flow
        CREATE_PROFILES_FUNC = run_profile_creation_with_clusters_flow
        logger.info("Using 'run_profile_creation_with_clusters_flow' from create_profiles_with_clusters.")
    except ImportError:
        logger.warning("Failed to import 'run_profile_creation_with_clusters_flow', trying 'create_or_update_profiles'.")
        from create_profiles_with_clusters import create_or_update_profiles
        from system_provider import get_system # Needed for the old signature
        CREATE_PROFILES_FUNC = lambda: create_or_update_profiles(get_system()) # Adapt the call
        logger.info("Using 'create_or_update_profiles' from create_profiles_with_clusters.")

except ImportError as e:
    logger.exception(f"Halt: Failed to import one or more necessary functions for the init flow: {e}")
    sys.exit(1)

def main_flow():
    logger.info("🚀 Starting full data initialization and processing flow...")

    # Default parameters (can be overridden by CLI args or config files in a more complex setup)
    # These path parameters assume `init_system_data.py` is run from the repository root.
    # Adjust paths if your files (categories.jsonl, output files) are elsewhere.
    # Data files for vector_db
    vd_jsonl_files = ["data/out_10_nodup_nonull.jsonl", "data/out_12_nodup_nonull.jsonl"]
    # General config files - ensure these paths are correct relative to where this script is run
    # or use absolute paths / more robust path configuration.
    categories_file_for_system = "data/categories.jsonl" # Used by get_system() via various steps
    qdrant_host_for_system = "localhost"
    qdrant_port_for_system = 6333

    # Specific files for create_profiles_with_clusters
    cpc_category_mappings_file = 'data/categories_map.json'
    cpc_output_profiles_file = 'supplier_profiles_with_clusters.json'


    # --- Step 1: Create/Update Vector Database ---
    logger.info("\n===== STEP 1: Створення/оновлення векторної бази =====")
    try:
        # create_optimized_vector_database uses get_system() internally and passes these params
        # to get_system() on its first call if the system isn't initialized yet.
        create_optimized_vector_database(
            jsonl_files=vd_jsonl_files,
            categories_file=categories_file_for_system,
            qdrant_host=qdrant_host_for_system,
            qdrant_port=qdrant_port_for_system,
            # Other params for create_optimized_vector_database like collection_name, batch_size, etc.,
            # will use their defaults defined within that function.
        )
        logger.info("✅ STEP 1: Завершено.")
    except Exception as e:
        logger.exception("❌ STEP 1: Помилка під час створення/оновлення векторної бази.")
        # Depending on severity, you might choose to exit: sys.exit(1)

    # --- Step 2: Enable Indexing ---
    logger.info("\n===== STEP 2: Увімкнення індексації у векторній базі =====")
    try:
        # This function directly interacts with Qdrant, does not use TenderAnalysisSystem
        enable_indexing_with_timeout(
            # Uses defaults: collection_name="tender_vectors", host="localhost", port=6333
        )
        logger.info("✅ STEP 2: Завершено (індексація може тривати у фоні).")
    except Exception as e:
        logger.exception("❌ STEP 2: Помилка під час увімкнення індексації.")

    # --- Step 3: Build Cache of Supplier Data (all_data_cache.pkl) ---
    logger.info("\n===== STEP 3: Створення/оновлення кешу даних постачальників (all_data_cache.pkl) =====")
    try:
        # run_build_cache uses get_system() internally.
        # It passes its categories_file, qdrant_host, qdrant_port to get_system if it's the first call.
        run_build_cache(
            categories_file=categories_file_for_system,
            qdrant_host=qdrant_host_for_system,
            qdrant_port=qdrant_port_for_system,
            # force_rebuild_cache uses its default (False)
        )
        logger.info("✅ STEP 3: Завершено.")
    except Exception as e:
        logger.exception("❌ STEP 3: Помилка під час створення/оновлення кешу all_data_cache.pkl.")
        # sys.exit(1)

    # --- Step 4: Create/Update Profiles with Clusters ---
    logger.info("\n===== STEP 4: Створення/оновлення профілів з кластерами та конкурентами =====")
    try:
        if CREATE_PROFILES_FUNC.__name__ == "run_profile_creation_with_clusters_flow":
            # This refactored function handles get_system() internally.
            # It passes its sys_categories_file etc. to get_system if it's the first call.
             CREATE_PROFILES_FUNC(
                sys_categories_file=categories_file_for_system,
                sys_qdrant_host=qdrant_host_for_system,
                sys_qdrant_port=qdrant_port_for_system,
                category_mappings_filepath=cpc_category_mappings_file,
                output_profiles_filepath=cpc_output_profiles_file,
                # force_rebuild_all_data_cache_from_db uses its default (False)
            )
        else: # Fallback to original create_or_update_profiles that needs system passed
            logger.info(f"Calling adapted {CREATE_PROFILES_FUNC.__name__}...")
            CREATE_PROFILES_FUNC() # Lambda wrapper calls get_system()

        logger.info("✅ STEP 4: Завершено.")
    except Exception as e:
        logger.exception("❌ STEP 4: Помилка під час створення/оновлення профілів з кластерами.")

    logger.info("\n🎉🎉🎉 Всі етапи ініціалізації та обробки даних завершено! 🎉🎉🎉")

if __name__ == "__main__":
    # CLI argument parsing can be added here to override defaults if needed
    # For example:
    # import argparse
    # parser = argparse.ArgumentParser(description="Run the full data initialization pipeline.")
    # parser.add_argument('--force-rebuild-all', action='store_true', help='Force rebuild for all relevant steps')
    # args = parser.parse_args()
    # if args.force_rebuild_all:
    #   logger.info("Force rebuild for all steps activated via CLI.")
    #   # You would then pass force flags to relevant functions.

    main_flow()