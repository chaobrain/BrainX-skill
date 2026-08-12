import unittest

import numpy as np

from sleep_replay import classify_replay, direction_scores, recall_scores


class ReplayAnalysisTests(unittest.TestCase):
    def test_forward_replay_direction_uses_temporal_order(self):
        events = np.zeros((12, 1, 4), dtype=bool)
        events[[1, 4, 7, 10], 0, np.arange(4)] = True
        forward, backward = direction_scores(events)
        self.assertEqual(forward.tolist(), [3])
        self.assertEqual(backward.tolist(), [0])
        self.assertEqual(classify_replay(forward, backward), "forward")

    def test_backward_replay_direction_uses_temporal_order(self):
        events = np.zeros((12, 1, 4), dtype=bool)
        events[[10, 7, 4, 1], 0, np.arange(4)] = True
        forward, backward = direction_scores(events)
        self.assertEqual(forward.tolist(), [0])
        self.assertEqual(backward.tolist(), [3])
        self.assertEqual(classify_replay(forward, backward), "backward")

    def test_recall_requires_an_ordered_prefix(self):
        events = np.zeros((12, 3, 4), dtype=bool)
        events[[1, 3, 5, 7], 0, np.arange(4)] = True
        events[[1, 3, 5], 1, np.arange(3)] = True
        events[[1, 7, 5, 3], 2, np.arange(4)] = True
        np.testing.assert_allclose(
            recall_scores(events), [1.0, 2.0 / 3.0, 1.0 / 3.0]
        )


if __name__ == "__main__":
    unittest.main()
