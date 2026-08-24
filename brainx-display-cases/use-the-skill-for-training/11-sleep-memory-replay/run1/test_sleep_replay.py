import numpy as np

import sleep_replay as model


def test_learning_protocol_visits_places_in_order():
    cue = np.asarray(model.learning_protocol().to_decimal(model.u.nA))
    population = cue.reshape(cue.shape[0], model.N_PLACES, model.CELLS_PER_PLACE)
    active = population.mean(axis=-1).argmax(axis=-1)
    first_visit = [int(np.flatnonzero(active == place)[0]) for place in range(4)]
    assert first_visit == sorted(first_visit)


def test_matched_sleep_current_is_identical_within_pairs():
    current = np.asarray(model.sleep_intrinsic_current().to_decimal(model.u.nA))
    np.testing.assert_array_equal(current[:, 0::2], current[:, 1::2])


def test_replay_detector_distinguishes_direction():
    spikes = np.zeros((48, 2, model.N_NEURONS), dtype=bool)
    for index, place in enumerate(range(4)):
        start = place * model.CELLS_PER_PLACE
        spikes[index * 6, 0, start : start + 3] = True
    for index, place in enumerate(range(3, -1, -1)):
        start = place * model.CELLS_PER_PLACE
        spikes[index * 6, 1, start : start + 3] = True
    events = model.detect_replays(spikes)
    assert (0, 0, "forward") in events
    assert (1, 0, "backward") in events


def test_end_to_end_matched_replay_intervention():
    result = model.run_experiment()
    np.testing.assert_array_equal(
        result.learned_weights[0::2], result.learned_weights[1::2]
    )
    assert any(lane % 2 == 0 for lane, _, _ in result.replay_events)
    assert all(lane % 2 == 0 for lane, _, _ in result.replay_events)
    assert result.recall_scores[0::2].mean() > result.recall_scores[1::2].mean()
