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
    assert "task_id=" in output
    assert "final_score=" in output

    assert "step_index=" in output
    assert "action_type=" in output
    assert "action_payload=" in output
    assert "observation_summary=" in output
    assert "reward=" in output
    assert "done=" in output

    assert "action_type=assign_interviewer" in output
    assert "action_type=summarize_fit" in output
    assert "action_type=choose_compensation_band" in output

