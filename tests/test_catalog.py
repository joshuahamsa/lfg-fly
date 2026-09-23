import json

from lfg_fly.teacher import catalog as K


def _fake_get(layers_ok):
    def get(url, timeout=30):
        if "/api/rarity?body=" in url:
            body = url.split("body=")[1]
            slots = {"Head": [{"value": f"{body}-hat", "odds_pct": 50.0, "enabled": True},
                              {"value": "None", "odds_pct": 50.0, "enabled": True}],
                     "Body": [{"value": f"{body}-body", "odds_pct": 100.0, "enabled": True}]}
            return 200, json.dumps({"body": body, "slots": slots}).encode()
        if "/api/layer?" in url:
            ok = any(v in url for v in layers_ok)
            return (200, b"PNGDATA") if ok else (404, b"")
        return 404, b""

    return get


def test_catalog_unions_bodies_filters_by_layer_and_adds_none(tmp_path):
    get = _fake_get(layers_ok=["male-hat", "female-hat", "male-body"])
    cat = K.build_catalog("http://lfg", tmp_path, body="male", get=get)
    assert cat.values["Head"] == ["female-hat", "male-hat", "None"]
    assert cat.values["Body"] == ["male-body"]  # only the hero's own body class
    assert "None" in cat.values["Accessory"] and cat.values["Accessory"] == ["None"]
    assert cat.odds["Head"] == {"male-hat": 50.0, "None": 50.0}
    assert K.layer_cache_path(tmp_path, "male", "Head", "male-hat").read_bytes() == b"PNGDATA"
    assert K.layer_cache_path(tmp_path, "male", "Head", "ape-hat").read_bytes() == b""  # cached 404


def test_catalog_json_roundtrip(tmp_path):
    cat = K.build_catalog("http://lfg", tmp_path, get=_fake_get(["male"]))
    assert K.Catalog.from_json(cat.to_json()) == cat


def test_cached_layers_are_not_refetched(tmp_path):
    calls = []
    base = _fake_get(["male-hat", "male-body"])

    def counting(url, timeout=30):
        calls.append(url)
        return base(url, timeout)

    K.build_catalog("http://lfg", tmp_path, get=counting)
    n = sum("/api/layer?" in u for u in calls)
    K.build_catalog("http://lfg", tmp_path, get=counting)
    assert sum("/api/layer?" in u for u in calls) == n
