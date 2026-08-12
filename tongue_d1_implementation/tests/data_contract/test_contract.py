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

def test_non_strict_reports_review_without_error():
    e,w=validate_contract("ontology/tongue_phenotype_v1.yaml","ontology/mappings",False)
    assert e==[]
    assert any("needs_review" in x for x in w)

def test_strict_blocks_review():
    e,w=validate_contract("ontology/tongue_phenotype_v1.yaml","ontology/mappings",True)
    assert any("needs_review" in x for x in e)

def test_na_semantics():
    assert normalize_na("NA") is None
    assert normalize_na("") is None
    assert normalize_na(0)==0
    assert normalize_na(1)==1

def test_tonguexpert_l2_is_pseudo():
    d=yaml.safe_load(Path("ontology/mappings/tonguexpert_v1.yaml").read_text(encoding="utf-8"))
    assert d["sources"]["L2"]["label_source"]=="model_prediction"
    assert d["sources"]["L2"]["supervision_tier"]=="pseudo"
