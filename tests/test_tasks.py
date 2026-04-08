from app.tasks import get_task_registry


def test_task_registry_has_minimum_tasks_and_difficulties():
    tasks = get_task_registry()
    assert len(tasks) >= 3

    difficulties = {task.difficulty.value for task in tasks}
    assert {"easy", "medium", "hard"}.issubset(difficulties)


def test_task_shapes_are_valid():
    tasks = get_task_registry()
    for task in tasks:
        assert task.task_id
        assert task.max_steps >= 3
        assert len(task.candidate_ids) >= 3
        assert len(task.expected_shortlist) >= 1
        assert task.expected_offer_candidate_id

