from wpu.engines.scheduler import ExecutionPath, Scheduler, SchedulerMetrics


def test_scheduler_thresholds() -> None:
    scheduler = Scheduler()
    assert scheduler.choose_path(SchedulerMetrics(1, 1.0, 1, 1, 100)).path == ExecutionPath.SPARSE
    assert scheduler.choose_path(SchedulerMetrics(1, 2.5, 2, 1, 100)).path == ExecutionPath.HYBRID
    assert scheduler.choose_path(SchedulerMetrics(4, 3.0, 2, 3, 100)).path == ExecutionPath.DENSE
