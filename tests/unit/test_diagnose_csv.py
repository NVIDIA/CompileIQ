import importlib.util
from pathlib import Path

import pandas as pd


def _load_diagnose_csv():
    script_path = (
        Path(__file__).parents[2]
        / "agent-skills"
        / "compileiq-debug"
        / "scripts"
        / "diagnose_csv.py"
    )
    spec = importlib.util.spec_from_file_location("diagnose_csv", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnose_flags_generation_csv_without_score_column():
    diagnose_csv = _load_diagnose_csv()
    df = pd.DataFrame({"generation": [0], "foo": [1]})

    diagnosis = diagnose_csv.diagnose(df)

    assert diagnosis.flags == ["NO_SCORE_COLUMN"]
    assert diagnosis.summary.empty
    assert "score_1" in diagnosis.notes[0]


def test_diagnose_accepts_legacy_score_column_alias():
    diagnose_csv = _load_diagnose_csv()
    df = pd.DataFrame({"generation": [0], "score": [1.0]})

    diagnosis = diagnose_csv.diagnose(df)

    assert diagnosis.flags == ["HEALTHY"]
