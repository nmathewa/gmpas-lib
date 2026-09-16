"""The background job runner both long reads go through."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from gmpas.jobs import Cancelled, Jobs


def _slow(started=None, release=None, value=None, steps=3):
    """A job that reports progress and can be held open by the test."""
    def work(progress, cancel, publish=None):
        if started is not None:
            started.set()
        for i in range(steps):
            if release is not None and not release.wait(5):
                raise AssertionError("release never came")
            if cancel.is_set():
                raise Cancelled()
            progress(i + 1, steps)
        return np.arange(4.0) if value is None else value
    return work


def _wait_done(jobs, key, total, work, tries=200):
    for _ in range(tries):
        state = jobs.progress(key, total, work)
        if state["state"] != "running":
            return state
        threading.Event().wait(0.01)
    raise AssertionError("job never finished")


def test_progress_returns_at_once_and_the_result_arrives_later():
    """The call must return while the read is still going -- the whole point
    is that the page is answered now and polls for the rest."""
    jobs = Jobs()
    started, release = threading.Event(), threading.Event()
    work = _slow(started, release)
    state = jobs.progress("k", 3, work)
    assert started.wait(5)                           # it really did start
    assert state["state"] == "running" and state["progress"] == [0, 3]
    release.set()
    assert _wait_done(jobs, "k", 3, work)["state"] == "done"
    assert np.array_equal(jobs.peek("k"), np.arange(4.0))


def test_a_finished_job_is_served_from_the_cache_without_running_again():
    jobs = Jobs()
    runs = []

    def work(progress, cancel, publish=None):
        runs.append(1)
        return np.zeros(2)

    _wait_done(jobs, "k", 1, work)
    for _ in range(3):
        assert jobs.progress("k", 1, work)["state"] == "done"
    assert len(runs) == 1


def test_a_failure_is_remembered_rather_than_retried_on_every_poll():
    """The page polls twice a second; a read that raises must not raise once
    per poll for as long as the page is open."""
    jobs = Jobs()
    runs = []

    def work(progress, cancel, publish=None):
        runs.append(1)
        raise ValueError("no such band")

    for _ in range(5):
        state = jobs.progress("k", 1, work)
        if state["state"] == "error":
            break
        threading.Event().wait(0.02)
    assert state["state"] == "error" and "no such band" in state["error"]
    jobs.progress("k", 1, work)
    assert len(runs) == 1


def test_a_new_request_cancels_the_one_already_reading():
    jobs = Jobs()
    started, release = threading.Event(), threading.Event()
    jobs.progress("first", 3, _slow(started, release))
    assert started.wait(5)
    jobs.progress("second", 1, lambda progress, cancel, publish=None: np.zeros(1))
    release.set()
    for _ in range(200):
        if jobs.running() == 0:
            break
        threading.Event().wait(0.01)
    assert jobs.peek("first") is None                # cancelled, nothing cached
    assert _wait_done(jobs, "second", 1,
                      lambda p, c, pub=None: np.zeros(1))["state"] == "done"


def test_a_waiting_export_keeps_its_job_from_being_cancelled():
    jobs = Jobs()
    started, release = threading.Event(), threading.Event()
    out = {}

    def export():
        out["value"] = jobs.result("first", 3, _slow(started, release))

    thread = threading.Thread(target=export)
    thread.start()
    assert started.wait(5)
    jobs.progress("second", 1, lambda progress, cancel, publish=None: np.zeros(1))  # would cancel
    release.set()
    thread.join(10)
    assert np.array_equal(out["value"], np.arange(4.0))


def test_stopping_cancels_everything_and_waits_for_it():
    jobs = Jobs()
    started, release = threading.Event(), threading.Event()
    jobs.progress("k", 3, _slow(started, release))
    assert started.wait(5)
    release.set()
    jobs.stop(timeout=10)
    assert jobs.running() == 0


def test_results_are_bounded_by_bytes_like_every_other_cache():
    jobs = Jobs(budget=1024)
    assert jobs.cache.budget == 1024
    _wait_done(jobs, "small", 1, lambda p, c, pub=None: np.zeros(8))
    assert jobs.peek("small") is not None


def test_the_result_call_raises_what_the_work_raised():
    jobs = Jobs()

    def work(progress, cancel, publish=None):
        raise KeyError("t2m")

    with pytest.raises(ValueError, match="t2m"):
        jobs.result("k", 1, work)
