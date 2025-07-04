import logging
from tender_analysis_system import TenderAnalysisSystem

_current_system: TenderAnalysisSystem | None = None
_system_initialized_successfully = False

logger = logging.getLogger(__name__)

# Configure logger for this module specifically if not configured by TenderAnalysisSystem
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_system(categories_file: str | None = "categories.jsonl",
               qdrant_host: str = "localhost",
               qdrant_port: int = 6333) -> TenderAnalysisSystem:
    """
    Provides a singleton instance of TenderAnalysisSystem.

    The system is initialized only on the first call.
    Subsequent calls return the existing instance.
    Configuration parameters are used only during the first initialization.
    """
    global _current_system
    global _system_initialized_successfully

    if _current_system is None:
        logger.info("System provider: No existing system instance found. Creating and initializing a new one.")
        try:
            # Suppress verbose logging from sentence_transformers during initialization
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)

            _current_system = TenderAnalysisSystem(
                categories_file=categories_file,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port
            )
            if _current_system.initialize_system():
                logger.info("System provider: TenderAnalysisSystem initialized successfully.")
                _system_initialized_successfully = True
            else:
                logger.error("System provider: TenderAnalysisSystem initialization failed. Subsequent calls to get_system will raise an error.")
                # Keep _current_system as None or handle error appropriately
                # For now, let's allow it to be set, but check _system_initialized_successfully
                # raise RuntimeError("System provider: Failed to initialize TenderAnalysisSystem.")
        except Exception as e:
            logger.exception(f"System provider: Exception during TenderAnalysisSystem instantiation or initialization: {e}")
            _current_system = None # Ensure it's None if init fails badly
            raise
        finally:
            # Restore logging levels if needed, though TenderAnalysisSystem might also set them
            logging.getLogger("sentence_transformers").setLevel(logging.INFO)
            logging.getLogger("transformers").setLevel(logging.INFO)


    if not _system_initialized_successfully and _current_system is not None:
        # This case means initialization was attempted, failed, but _current_system object exists.
        # Or if _current_system is None, it means a severe error happened above.
        raise RuntimeError("System provider: TenderAnalysisSystem was not initialized successfully on a previous attempt.")

    if _current_system is None:
        # This means an exception occurred above, and _current_system was reset or never set.
        raise RuntimeError("System provider: Unable to provide TenderAnalysisSystem instance due to critical initialization error.")

    return _current_system

def is_system_initialized() -> bool:
    """
    Checks if the system has been initialized successfully.
    """
    global _system_initialized_successfully
    return _system_initialized_successfully

if __name__ == '__main__':
    # Example usage and test
    print("Attempting to get system instance (1st call):")
    try:
        system1 = get_system()
        print(f"System 1 instance: {system1}, Initialized: {system1.is_initialized}")

        print("\nAttempting to get system instance (2nd call):")
        system2 = get_system()
        print(f"System 2 instance: {system2}, Initialized: {system2.is_initialized}")

        print(f"\nAre system1 and system2 the same object? {'Yes' if system1 is system2 else 'No'}")
        print(f"Is system initialized according to provider? {is_system_initialized()}")

        # Test re-initialization attempt (it shouldn't re-initialize)
        print("\nAttempting to get system with different params (should return existing):")
        system3 = get_system(categories_file="different.jsonl")
        print(f"System 3 instance: {system3}, Categories file: {system3.categories_file}")
        print(f"Is system1 and system3 the same object? {'Yes' if system1 is system3 else 'No'}")


    except Exception as e:
        print(f"An error occurred during test: {e}")
        logger.exception("Error during system_provider test.")

    # Test case for failed initialization
    print("\n--- Test Case: Simulating Failed Initialization ---")
    _current_system = None # Reset for testing
    _system_initialized_successfully = False

    class MockTenderAnalysisSystem(TenderAnalysisSystem):
        def initialize_system(self):
            print("MockTenderAnalysisSystem: Simulating initialization failure.")
            self.is_initialized = False # Explicitly set, though super().initialize_system might do this
            return False

    original_system_class = TenderAnalysisSystem # Save original

    try:
        # Monkey patch TenderAnalysisSystem for this test block
        import sys
        # This is a bit tricky; we need to make get_system use the mock.
        # The cleanest way would be to pass the class to get_system,
        # but for this test, we'll rely on it picking up the changed global name if it were imported differently.
        # A direct patch is harder as TenderAnalysisSystem is imported directly.
        # For simplicity of this self-contained test, we assume we can influence the class it uses.
        # This test might not work perfectly without more advanced patching like unittest.mock.

        # Let's try to simulate the failure by directly manipulating the global var used by get_system
        # This is not ideal, but for a quick test:

        # This is a conceptual test. A real test would use unittest.mock.patch
        # to replace TenderAnalysisSystem within the system_provider module's scope.
        print("Conceptual test: If TenderAnalysisSystem() call inside get_system() used a mock that failed init:")

        # Simulate get_system trying to initialize a faulty system
        _current_system = TenderAnalysisSystem() # A real instance
        _current_system.initialize_system = lambda: False # Monkey patch its method

        if not _current_system.initialize_system():
             _system_initialized_successfully = False
             logger.error("Simulated initialization failure for test.")

        try:
            system_fail = get_system() # This call will now use the pre-failed _current_system
        except RuntimeError as e:
            print(f"Caught expected error: {e}")

    except Exception as e:
        print(f"Error in failed init test: {e}")
    finally:
        # Restore
        _current_system = None
        _system_initialized_successfully = False
        # TenderAnalysisSystem = original_system_class # Restore (if we had truly patched it)

    print("\nTest completed.")
