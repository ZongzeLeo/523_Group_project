import numpy as np
import pytest

from geometry import GeometryError
from geometry.publisher import measure_publisher
from geometry.publisher_profiles import publisher_profile, measure_profiles


def test_original_profile_reuse_matches_full_notebook_and_cannot_mutate_cache():
    yy, xx = np.indices((85, 65))
    frames = {}
    for view, width in (("2CH", 18), ("4CH", 21)):
        for phase, radius in (("ED", 32), ("ES", 25)):
            mask = ((xx-32)**2/width**2+(yy-40)**2/radius**2 <= 1) & (yy <= 58)
            frames[f"{view}_{phase}"] = {"mask": mask, "geometry": {"spacing_xy_mm": [.8, .9]}}
    original = measure_publisher(frames)
    profiles = {k: publisher_profile(f) for k, f in frames.items()}
    saved = {k: v["diameters_mm"].copy() for k, v in profiles.items()}
    for _ in range(2):
        result = measure_profiles(profiles)
        for field in ("edv_ml", "esv_ml", "ef_pct", "ef_before_final_rounding_pct"):
            assert result[field] == original[field]
    for key in profiles:
        np.testing.assert_array_equal(profiles[key]["diameters_mm"], saved[key])
    profiles["2CH_ED"] = None
    with pytest.raises(GeometryError):
        measure_profiles(profiles)
