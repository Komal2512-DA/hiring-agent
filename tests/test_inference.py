import io
from contextlib import redirect_stdout

import inference


def test_inference_stdout_structure():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = inference.main([])

    assert exit_code == 0
    output = buffer.getvalue()

    assert "[START]" in output
    assert "[STEP]" in output
    assert "[END]" in output
    assert "task=" in output
    assert "env=" in output
    assert "model=" in output
    assert "step=" in output
    assert "action=" in output
    assert "reward=" in output
    assert "done=" in output
    assert "error=" in output
    assert "success=" in output
    assert "steps=" in output
    assert "score=" in output
    assert "rewards=" in output

    assert "action=shortlist_candidates(" in output
    assert "action=advance_stage(" in output
    assert "action=finalize_decision(" in output

