ML + EKF source code

Run from the submission's main directory:

    bash code/RUN_ML_EKF.sh

Main modules

ml_ekf/core.py       Causal features, resampling and corrected EKF update.
ml_ekf/networks.py   Feedforward network, scaling and training windows.
ml_ekf/runner.py     Sequential prediction from voltage, current and temperature.
ml_ekf/dataset.py    Recording groups and training/validation separation.
ml_ekf/train.py      Training and validation-based correction selection.
ml_ekf/reproduce_final.py
                    Benchmark evaluation, figures and PDF report.
physics_backend/    Circuit parameter tables and the baseline EKF.
run_submission.py   Input verification and execution of the full evaluation.

artifacts/fnn.pt contains the trained network. fnn.json contains its feature
scaler and correction settings. The physics calibration is stored in
physics_backend/models/physics_selection.json.

The code.zip archive contains the same source and dependency requirements.
It is provided for distribution; the launcher uses the extracted source here.
Retraining should be performed in a separate working copy.
