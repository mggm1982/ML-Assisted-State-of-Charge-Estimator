Evaluation results

fresh_evaluation_metrics.csv contains one row for each of the ten benchmark
recordings. ML_EKF columns refer to the combined ML + EKF model. Errors are
expressed in SOC percentage points. The improvement column compares each
recording with the EKF baseline operating without neural corrections.

figures/ contains three summary plots and four representative SOC/error plots.
Green bars indicate lower RMSE with ML + EKF; red bars indicate higher RMSE.
predictions/ contains the ten numerical replays used to calculate the metrics.
ML_EKF_Evaluation_Report.pdf presents the results and all seven figures.

REPRODUCTION_PROVENANCE.json records the model, input and source hashes.
reproduction/ contains the execution log, unit-test log and run status.

Running bash code/RUN_ML_EKF.sh from the project root loads the trained model
and regenerates predictions, metrics, plots and the evaluation report from
raw MAT measurements. This command performs evaluation; it does not retrain
the model.

ML + EKF improves RMSE on eight of the ten recordings. Mean recording-level
RMSE decreases from 0.5863407 to 0.4516111 SOC percentage points, a reduction
of 22.98%. Both UDDS recordings have higher RMSE with ML + EKF. These benchmark
recordings were examined during development, so the results do not represent
an independently unseen test set.
