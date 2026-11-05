import unittest
from pathlib import Path
import numpy as np
from ml_ekf.core import frozen_physics, one_hz, read_sensor_csv, CorrectedEKF
from ml_ekf.runner import run

ROOT = Path(__file__).resolve().parents[1]


class FinalModelChecks(unittest.TestCase):
    def setUp(self):
        self.samples = one_hz(
            read_sensor_csv(ROOT / "examples/measurements_only_US06.csv.gz")
        )[:100]

    def test_model_and_physics_present(self):
        self.assertTrue((ROOT / "artifacts/fnn.pt").is_file())
        self.assertTrue(
            (ROOT / "physics_backend/models/physics_selection.json").is_file()
        )
        battery, cfg = frozen_physics()
        self.assertIn("capacity_Ah", battery)

    def test_causal_replay_finite(self):
        out = run(
            self.samples,
            0.8,
            artifact=ROOT / "artifacts/fnn.pt",
            alpha_voltage=0.5,
            beta_soc=0.5,
        )
        self.assertEqual(len(out["soc_hybrid"]), len(self.samples))
        self.assertTrue(np.isfinite(out["soc_hybrid"]).all())
        self.assertTrue(np.all((out["soc_hybrid"] >= 0) & (out["soc_hybrid"] <= 1)))
        # ML suppressed in initial warmup
        self.assertTrue(np.array_equal(out["soc_ekf"][:60], out["soc_hybrid"][:60]))

    def test_zero_correction_equivalent_to_calibrated_ekf(self):
        from physics import EKF

        b, c = frozen_physics()
        a = EKF(b, c, 0.8)
        z = CorrectedEKF(b, c, 0.8)
        for p in self.samples:
            np.testing.assert_array_equal(a.step(*p), z.step(*p, offset_V=0))


if __name__ == "__main__":
    unittest.main()
