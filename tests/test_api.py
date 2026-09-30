"""Tests for the inference API (uses a small untrained checkpoint)."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.models.factory import build_model
from src.utils.config import load_config


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    cfg = load_config()
    cfg["preprocessing"].update(denoise="none", clahe=False, ben_graham=False)
    model = build_model("mobilenetv3", 5, pretrained=False)
    torch.save({"model_state": model.state_dict(), "backbone": "mobilenetv3", "num_classes": 5,
                "class_names": cfg["data"]["class_names"],
                "config": {"preprocessing": cfg["preprocessing"], "image": cfg["image"]},
                "epoch": 1, "val_metrics": {"qwk": 0.5}}, tmp / "best.pt")
    (tmp / "card.json").write_text(json.dumps({"run": "test_run", "screening_threshold": 0.3,
                                               "test_metrics": {"accuracy": 0.8}}))
    from api.main import create_app
    with TestClient(create_app(tmp / "best.pt", tmp / "card.json")) as c:
        yield c


def fundus_png(brightness: int = 150) -> bytes:
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 130, (40, brightness // 2, brightness), -1)
    for x, y in [(120, 130), (170, 160), (150, 100)]:
        cv2.circle(img, (x, y), 3, (20, 20, 90), -1)
    img = cv2.add(img, np.random.default_rng(0).integers(0, 6, img.shape, dtype=np.uint8))
    return cv2.imencode(".png", img)[1].tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    m = r.json()["model"]
    assert m["run"] == "test_run" and m["screening_threshold"] == 0.3 and len(m["class_names"]) == 5


def test_predict_returns_complete_result(client):
    r = client.post("/predict", files={"file": ("eye.png", fundus_png(), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0 <= d["stage"] <= 4 and d["label"] in [p["label"] for p in d["probabilities"]]
    assert abs(sum(p["probability"] for p in d["probabilities"]) - 1) < 0.01
    assert d["refer"] == (d["dr_probability"] >= d["screening_threshold"])
    for key in ("model_input_png", "gradcam_png"):
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d[key]), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (224, 224, 3)


def test_good_image_has_no_quality_warning(client):
    r = client.post("/predict", files={"file": ("eye.png", fundus_png(), "image/png")})
    assert not [w for w in r.json()["warnings"] if "confidence" not in w]


def test_dark_image_gets_quality_warning(client):
    r = client.post("/predict", files={"file": ("dark.png", fundus_png(brightness=40), "image/png")})
    assert any("dark" in w for w in r.json()["warnings"])


def test_ordinary_photo_is_flagged(client):
    photo = np.full((200, 300, 3), 170, np.uint8)
    cv2.rectangle(photo, (50, 40), (250, 160), (30, 120, 200), -1)
    r = client.post("/predict", files={"file": ("cat.png", cv2.imencode(".png", photo)[1].tobytes(), "image/png")})
    assert any("does not look like a fundus" in w for w in r.json()["warnings"])


def test_non_fundus_image_is_flagged(client):
    blank = cv2.imencode(".png", np.zeros((200, 200, 3), np.uint8))[1].tobytes()
    r = client.post("/predict", files={"file": ("black.png", blank, "image/png")})
    assert r.status_code == 200
    assert any("retina" in w for w in r.json()["warnings"])


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_web_page_is_served(client):
    r = client.get("/app/")
    assert r.status_code == 200 and "Diabetic retinopathy grading" in r.text
