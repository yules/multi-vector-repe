import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import download_datasets


class DownloadDatasetsTests(unittest.TestCase):
    @patch("download_datasets.load_dataset")
    def test_download_and_save_datasets_saves_harmbench(self, load_dataset):
        dataset = Mock()
        load_dataset.return_value = dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            download_datasets.download_and_save_datasets(temp_dir)

            load_dataset.assert_called_once_with(
                "walledai/HarmBench", "standard", trust_remote_code=True
            )
            dataset.save_to_disk.assert_called_once_with(
                str(Path(temp_dir) / "harmbench")
            )

    @patch("download_datasets.load_dataset", side_effect=RuntimeError("network error"))
    def test_download_and_save_datasets_handles_download_failure(self, load_dataset):
        with tempfile.TemporaryDirectory() as temp_dir:
            download_datasets.download_and_save_datasets(temp_dir)

        load_dataset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
