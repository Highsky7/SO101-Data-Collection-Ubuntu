import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from d455_rgb import _encode_rgb_png, _prepare_output_dir


class RGBOutputTests(unittest.TestCase):
    def test_saved_png_preserves_rgb_channels_and_values(self):
        rgb = np.array([[[255, 0, 0], [0, 0, 255]], [[0, 255, 0], [23, 94, 177]]], dtype=np.uint8)
        with Image.open(io.BytesIO(_encode_rgb_png(rgb))) as image:
            np.testing.assert_array_equal(np.asarray(image), rgb)

    def test_existing_capture_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "d455_rgb.png"
            original.write_bytes(b"existing capture")
            with self.assertRaises(FileExistsError):
                _prepare_output_dir(Path(directory))
            self.assertEqual(original.read_bytes(), b"existing capture")


if __name__ == "__main__":
    unittest.main()
