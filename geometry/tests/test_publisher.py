import numpy as np
import pytest

from geometry import GeometryError
from geometry.publisher import measure_publisher


def test_publisher_rejects_incomplete_and_empty_masks():
    with pytest.raises(GeometryError, match="four"):
        measure_publisher({})
    frames = {f"{view}_{phase}": {"mask": np.zeros((32, 32)), "geometry": {"spacing_xy_mm": [1, 1]}}
              for view in ("2CH", "4CH") for phase in ("ED", "ES")}
    with pytest.raises(GeometryError, match="empty"):
        measure_publisher(frames)
