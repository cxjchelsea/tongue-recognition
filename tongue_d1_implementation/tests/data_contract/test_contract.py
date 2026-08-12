from tongue_data.ontology import Ontology
from tongue_data.validators import validate_contract
from tongue_data.utils import normalize_na
from pathlib import Path
import yaml

def test_ontology_valid():
    o=Ontology("ontology/tongue_phenotype_v1.yaml")
    assert o.validate()==[]
    assert o.has_task("features.crack.present")
    assert o.has_label("features.crack.present",True)

def test_non_strict_has_no_needs_review():
    # 三项待确认语义已拍板后，契约中不应再残留 needs_review
    e,w=validate_contract("ontology/tongue_phenotype_v1.yaml","ontology/mappings",False)
    assert e==[]
    assert not any("needs_review" in x for x in w)

def test_strict_contract_passes():
    e,w=validate_contract("ontology/tongue_phenotype_v1.yaml","ontology/mappings",True)
    assert e==[]
    assert not any("needs_review" in x for x in e)

def test_resolved_semantic_mappings():
    # DSCT：0/1 为裂纹轻重，而非有/无
    dsct=yaml.safe_load(Path("ontology/mappings/dsct_v1.yaml").read_text(encoding="utf-8"))
    assert dsct["mappings"]["0"]["canonical_label"]=="mild"
    assert dsct["mappings"]["1"]["canonical_label"]=="severe"
    # TonguExpert dark 独立映射，不等于 purple
    tx=yaml.safe_load(Path("ontology/mappings/tonguexpert_v1.yaml").read_text(encoding="utf-8"))
    assert tx["mappings"]["L1.labels_zhi.dark"]["canonical_label"]=="dark"
    assert tx["mappings"]["L2.zhi_label.dark"]["canonical_label"]=="dark"
    # TMC 滑苔在 V1 排除
    tmc=yaml.safe_load(Path("ontology/mappings/tmc_v1.yaml").read_text(encoding="utf-8"))
    assert tmc["mappings"]["huataishe"]["status"]=="excluded"

def test_na_semantics():
    assert normalize_na("NA") is None
    assert normalize_na("") is None
    assert normalize_na(0)==0
    assert normalize_na(1)==1

def test_tonguexpert_l2_is_pseudo():
    d=yaml.safe_load(Path("ontology/mappings/tonguexpert_v1.yaml").read_text(encoding="utf-8"))
    assert d["sources"]["L2"]["label_source"]=="model_prediction"
    assert d["sources"]["L2"]["supervision_tier"]=="pseudo"
