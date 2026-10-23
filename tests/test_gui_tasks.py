import threading
import time
import pytest

from gui_tasks import LatestTaskRunner, SearchBelief, SearchRandom, SearchCancelled


def wait_result(runner, token):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        result = runner.poll(token)
        if result is not None:
            return result
        time.sleep(.005)
    raise AssertionError('后台任务未完成')


def test_new_position_cancels_search_and_returns_only_latest_result():
    runner = LatestTaskRunner()
    started = threading.Event()

    def slow(rng):
        started.set()
        while True:
            rng.randrange(100)
            time.sleep(.001)

    old = runner.submit(slow)
    assert started.wait(1)
    new = runner.submit(lambda rng: ('latest', rng.randrange(1)))
    result = wait_result(runner, new)
    assert result.value == ('latest', 0)
    assert runner.poll(old) is None
    runner.close()
    runner._thread.join(1)
    assert not runner._thread.is_alive()


def test_pending_jobs_are_coalesced_and_errors_are_reported():
    runner = LatestTaskRunner()
    started, release = threading.Event(), threading.Event()
    ran = []

    def loading(rng):
        started.set()
        release.wait(2)
        rng.random()

    runner.submit(loading)
    assert started.wait(1)
    runner.submit(lambda rng: ran.append('superseded'))
    newest = runner.submit(lambda rng: ran.append('newest'))
    release.set()
    wait_result(runner, newest)
    assert ran == ['newest']
    token = runner.submit(lambda rng: 1 / 0)
    assert 'ZeroDivisionError' in wait_result(runner, token).error
    runner.close()


def test_cancelled_root_determinization_with_no_random_sampling():
    import jieqi as jq
    import ismcts as im
    state = jq.GameState.empty(with_pools=False)
    state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    state.place_piece(9, 3, jq.TYPE_KING, jq.RED, 0)
    rng = SearchRandom()
    view = SearchBelief(im.init_belief_for_game(state), rng).info_state(state, jq.RED)
    assert view.determinize(rng).board == state.board
    rng.cancelled.set()
    with pytest.raises(SearchCancelled):
        view.determinize(rng)
