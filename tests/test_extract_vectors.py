import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
import yaml

import extract_vectors


class ExtractVectorsTests(unittest.TestCase):
    def test_load_model_config_resolves_selected_model(self):
        config = {
            "model": {"name": "test-model"},
            "models": {"test-model": {"layer_idx": 7}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

            model_id, layer_idx = extract_vectors.load_model_config(config_path)

        self.assertEqual(model_id, "test-model")
        self.assertEqual(layer_idx, 7)

    def test_load_model_config_rejects_unknown_model(self):
        config = {
            "model": {"name": "missing-model"},
            "models": {"known-model": {"layer_idx": 7}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(yaml.safe_dump(config))

            with self.assertRaisesRegex(ValueError, "not defined"):
                extract_vectors.load_model_config(config_path)

    def test_get_latent_vector_sequential_normalizes_last_token(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        inputs = type("TokenizedInputs", (dict,), {
            "to": lambda self, device: self,
        })(input_ids=[[1, 2]])
        tokenizer.return_value = inputs

        model = Mock()
        hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        model.return_value.hidden_states = [hidden_state]

        vector = extract_vectors.get_latent_vector_sequential(
            "hello", tokenizer, model, torch.device("cpu"), 0
        )

        np.testing.assert_allclose(vector, np.array([0.6, 0.8]))
        tokenizer.apply_chat_template.assert_called_once()
        model.assert_called_once_with(input_ids=[[1, 2]])


if __name__ == "__main__":
    unittest.main()
