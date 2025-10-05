# adding parent directory to linking
import sys

sys.path.append("..")

from saving_model import save_model, load_model

import tempfile
import os
import torch
import torch.nn as nn
import pytest


from torch import jit


class TinyNet(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyNet()


def test_save_and_load_model_equivalence(model):
    """Sprawdza czy zapisany i odczytany model daje te same wyniki."""
    x = torch.randn(3, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")

        # --- zapis i odczyt ---
        save_model(model, path)
        loaded = load_model(path)

        # --- weryfikacja ---
        orig_out = model.eval()(x)
        new_out = loaded(x)

        # dokładne porównanie wyników
        assert torch.allclose(orig_out, new_out, atol=1e-6), "Wyniki modeli się różnią"
        assert isinstance(loaded, jit.ScriptModule)


def test_file_is_created(model):
    """Sprawdza czy plik został faktycznie zapisany."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        save_model(model, path)
        assert os.path.exists(path), "Plik modelu nie został zapisany"


def test_load_nonexistent_file():
    """Sprawdza czy ładowanie nieistniejącego pliku rzuca błąd."""
    with pytest.raises(ValueError):
        load_model("nonexistent_model.pt")


def test_invalid_input_to_save():
    """Sprawdza czy zapisanie obiektu niebędącego modelem rzuca błąd."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.pt")
        with pytest.raises(Exception):
            save_model("not_a_model", path)
