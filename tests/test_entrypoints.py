import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(script_name):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / script_name), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_training_entrypoint_help():
    result = run_help("train.py")

    assert result.returncode == 0
    assert "--device" in result.stdout
    assert "--num_channel" in result.stdout
    assert "--distributed" in result.stdout


def test_enhancement_entrypoint_help():
    for script_name in [
        "evaluate_folder.py",
        "evaluate_file.py",
        "evaluation.py",
        "single_file_evaluation.py",
        "streaming_input_evaluation.py",
    ]:
        result = run_help(script_name)

        assert result.returncode == 0, result.stderr
        assert "--device" in result.stdout
        assert "--num_channel" in result.stdout
        assert "--mask_mode" in result.stdout
        assert "--module" in result.stdout


def test_analysis_entrypoint_help():
    for script_name in ["ASR_eval.py", "Feature_evaluation.py"]:
        result = run_help(script_name)

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
