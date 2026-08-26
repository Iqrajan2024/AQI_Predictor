from pathlib import Path

from feast import FeatureStore


# ============================================================
# FEAST REPOSITORY PATH
# ============================================================

FEATURE_REPO = Path(__file__).resolve().parent


# ============================================================
# TEST: CONFIGURATION EXISTS
# ============================================================

def test_feature_store_config_exists():
    """Verify that feature_store.yaml exists."""

    config_file = FEATURE_REPO / "feature_store.yaml"

    assert config_file.exists(), (
        f"Feast configuration not found: {config_file}"
    )


# ============================================================
# TEST: FEATURE STORE LOADS
# ============================================================

def test_feature_store_loads():
    """Verify that Feast can initialize the FeatureStore."""

    config_file = FEATURE_REPO / "feature_store.yaml"

    assert config_file.exists(), (
        f"Feast configuration not found: {config_file}"
    )

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    assert store is not None