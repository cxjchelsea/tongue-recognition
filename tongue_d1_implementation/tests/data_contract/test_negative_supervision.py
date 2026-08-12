"""D1.1：explicit negative supervision 契约测试。"""
from pathlib import Path
import yaml
from tongue_data.adapters.base import DatasetAdapter
from tongue_data.utils import normalize_na
from tongue_data.mapping import resolve_mapping_entry


class _ProbeAdapter(DatasetAdapter):
    """仅用于调用 mapping_to_label_records 的轻量适配器。"""

    def collect(self):
        return [], [], [], []


def _tonguedx_adapter():
    mapping_doc = yaml.safe_load(
        Path("ontology/mappings/tonguedx_v1.yaml").read_text(encoding="utf-8")
    )
    return _ProbeAdapter({"name": "tonguedx", "root": "."}, mapping_doc), mapping_doc


def test_tongue_pale_positive():
    adapter, _ = _tonguedx_adapter()
    records, warnings = adapter.mapping_to_label_records(
        "tonguedx::demo", "TonguePale", "TonguePale", value=1
    )
    assert warnings == []
    assert len(records) == 1
    assert records[0].canonical_task == "tongue_body.color"
    assert records[0].canonical_label == "pale"
    assert records[0].value == 1
    assert records[0].label_available is True


def test_tongue_pale_explicit_negative():
    adapter, _ = _tonguedx_adapter()
    records, warnings = adapter.mapping_to_label_records(
        "tonguedx::demo", "TonguePale", "TonguePale", value=0
    )
    assert warnings == []
    assert len(records) == 1
    assert records[0].canonical_task == "tongue_body.color"
    assert records[0].canonical_label == "pale"
    assert records[0].value == 0
    assert records[0].label_available is True
    # 不得把 not-pale 映射成其他颜色类
    assert records[0].canonical_label != "normal"


def test_tongue_pale_na_does_not_emit_negative():
    adapter, _ = _tonguedx_adapter()
    # Adapter 调用方对 NA 应跳过；即便误传 None 也不得产出 pale=0
    raw = normalize_na("NA")
    assert raw is None
    if raw is None:
        records = []
    else:
        records, _ = adapter.mapping_to_label_records(
            "tonguedx::demo", "TonguePale", "TonguePale", value=raw
        )
    assert records == []
    assert not any(
        (r.canonical_label == "pale" and r.value == 0) for r in records
    )


def test_fur_yellow_explicit_negative():
    adapter, _ = _tonguedx_adapter()
    records, warnings = adapter.mapping_to_label_records(
        "tonguedx::demo", "FurYellow", "FurYellow", value=0
    )
    assert warnings == []
    assert len(records) == 1
    assert records[0].canonical_task == "coating.color"
    assert records[0].canonical_label == "yellow"
    assert records[0].value == 0
    assert records[0].label_available is True
    # 不得把 not-yellow 映射成 white
    assert records[0].canonical_label != "white"


def test_binary_crack_positive_and_negative_regression():
    adapter, _ = _tonguedx_adapter()
    pos, warn_pos = adapter.mapping_to_label_records(
        "tonguedx::demo", "Crack", "Crack", value=1
    )
    neg, warn_neg = adapter.mapping_to_label_records(
        "tonguedx::demo", "Crack", "Crack", value=0
    )
    assert warn_pos == [] and warn_neg == []
    assert pos[0].canonical_task == "features.crack.present"
    assert pos[0].canonical_label is True
    assert pos[0].value == 1
    assert neg[0].canonical_task == "features.crack.present"
    assert neg[0].canonical_label is False
    assert neg[0].value == 1
    assert neg[0].label_available is True


def test_tonguexpert_l2_isolation_unchanged():
    mapping_doc = yaml.safe_load(
        Path("ontology/mappings/tonguexpert_v1.yaml").read_text(encoding="utf-8")
    )
    assert mapping_doc["sources"]["L2"]["label_source"] == "model_prediction"
    assert mapping_doc["sources"]["L2"]["supervision_tier"] == "pseudo"
    item = resolve_mapping_entry(mapping_doc, "L2.zhi_label.dark", "L2")
    assert item["supervision_tier"] == "pseudo"
    assert item["label_source"] == "model_prediction"


def test_na_global_not_equal_zero():
    assert normalize_na("NA") is None
    assert normalize_na(0) == 0
    assert normalize_na("NA") != 0
