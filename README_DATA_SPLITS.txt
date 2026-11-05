Panasonic dataset and recording split

The complete Panasonic 18650PF dataset is included in:

    data/Panasonic 18650PF Data/

The experiment uses 16 selected recordings. Copies are organised in the following folders within data/:

Training data/
    25C Cycle 2 and Cycle 3
    10C Cycle 1 and Cycle 3

Training data/Validation data/
    Cycle 4 at 10C and 25C
    These two recordings are used for model selection, not gradient updates.

Testing data/
    10C: HWFET, NN, UDDS and US06
    25C: HWFTa, HWFTb, LA92, NN, UDDS and US06

The split folders contain actual MAT recordings. DATA_SPLIT_MANIFEST.csv lists their paths and SHA-256 hashes. Whole recordings are kept together rather than dividing individual time samples between subsets.

The two HWFT recordings at 25C are separate files. The 10C LA92 recording is excluded from this benchmark because of a recorded time discontinuity.

The testing folder contains previously examined benchmark recordings. It is used to report performance after model selection and is not an untouched, independent test set.

Complete dataset structure

The original filenames and nested directories are preserved. Paths relative to data/ include:

    Panasonic 18650PF Data/10degC/Drive Cycles/...
    Panasonic 18650PF Data/Panasonic 18650PF Data/25degC/Drive cycles/...

The repeated folder name in the 25C path is part of the original extraction. Keep this directory structure unchanged.

Running the evaluation

From the main project folder, run:

    bash code/RUN_ML_EKF.sh

The launcher uses the complete dataset when available. If it is absent, the launcher uses the selected MAT recordings in the split folders. In either case, it verifies the same 16 selected recording hashes before evaluating the ten benchmarks.

The command loads the saved trained model and EKF calibration, then calculates predictions, metrics and plots from the raw benchmark measurements. It does not retrain the model or use every recording in the complete dataset.

Evaluation outputs are written to Results/. The selected data source and input hashes are recorded in Results/REPRODUCTION_PROVENANCE.json.

Dataset source

Phillip Kollmeyer (2018), Panasonic 18650PF Li-ion Battery Data, Mendeley Data, version 1.
DOI: 10.17632/wykht8y7tg.1.