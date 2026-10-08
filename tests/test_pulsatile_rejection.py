import unittest

import numpy as np
import pandas as pd

from percept_parser.pulsatile_rejection import PulsatileArtifactRejector


class PulsatileArtifactRejectorTests(unittest.TestCase):
    def setUp(self):
        self.rejector = PulsatileArtifactRejector()
        self.sample_rate = 250

    def test_rejects_repetitive_pulses_at_60_bpm(self):
        data, _ = self.make_pulsatile_signal()

        cleaned = self.rejector.clean(pd.DataFrame({"Left": data}), self.sample_rate)

        rejected = cleaned["Left"].isna()
        self.assertGreater(rejected.sum(), len(data) * 0.01)
        self.assertLess(rejected.sum(), len(data) * 0.5)

    def test_subtracts_repetitive_pulses_without_creating_missing_samples(self):
        data, clean_signal = self.make_pulsatile_signal()

        cleaned = self.rejector.subtract(
            pd.DataFrame({"Left": data}), self.sample_rate
        )["Left"].to_numpy()

        trim = round(self.rejector.initial_trim_seconds * self.sample_rate)
        self.assertTrue(np.isnan(cleaned[:trim]).all())
        self.assertTrue(np.isfinite(cleaned[trim:]).all())
        start = trim + 2 * self.sample_rate
        end = -2 * self.sample_rate
        original_error = np.sqrt(
            np.mean((data[start:end] - clean_signal[start:end]) ** 2)
        )
        cleaned_error = np.sqrt(
            np.mean((cleaned[start:end] - clean_signal[start:end]) ** 2)
        )
        self.assertLess(cleaned_error, original_error * 0.7)

    def make_pulsatile_signal(self):
        rng = np.random.default_rng(42)
        sample_count = 40 * self.sample_rate
        clean_signal = rng.normal(0, 0.08, sample_count)
        data = clean_signal.copy()
        pulse = np.arange(-20, 21)
        pulse_shape = np.exp(-0.5 * (pulse / 4) ** 2) - 0.55 * np.exp(
            -0.5 * ((pulse - 7) / 5) ** 2
        )
        for center in range(self.sample_rate, sample_count, self.sample_rate):
            data[center - 20 : center + 21] += 3 * pulse_shape
        return data, clean_signal

    def test_leaves_nonperiodic_signal_unchanged(self):
        rng = np.random.default_rng(43)
        data = rng.normal(0, 1, 40 * self.sample_rate)
        frame = pd.DataFrame({"Left": data})

        cleaned = self.rejector.clean(frame, self.sample_rate)

        pd.testing.assert_frame_equal(
            cleaned, self.rejector.trim_start(frame, self.sample_rate)
        )

    def test_leaves_existing_missing_samples_untouched(self):
        data = np.zeros(40 * self.sample_rate)
        data[3000:3005] = np.nan
        frame = pd.DataFrame({"Left": data})

        cleaned = self.rejector.clean(frame, self.sample_rate)

        pd.testing.assert_frame_equal(
            cleaned, self.rejector.trim_start(frame, self.sample_rate)
        )

    def test_leaves_short_segments_unchanged(self):
        sample_count = 19 * self.sample_rate
        data = np.zeros(sample_count)
        pulse = np.arange(-20, 21)
        pulse_shape = np.exp(-0.5 * (pulse / 4) ** 2)
        for center in range(self.sample_rate, sample_count - 20, self.sample_rate):
            data[center - 20 : center + 21] += pulse_shape
        frame = pd.DataFrame({"Left": data})

        cleaned = self.rejector.clean(frame, self.sample_rate)

        pd.testing.assert_frame_equal(
            cleaned, self.rejector.trim_start(frame, self.sample_rate)
        )


if __name__ == "__main__":
    unittest.main()
