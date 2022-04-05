## How to run

Download [this emg dataset](https://data.mendeley.com/datasets/ckwc76xr2z/2), and put the /sEMG-dataset floder in the repository.

Find a BCICIV_2a_gdf eeg dataset. put it into the repository.

Use the get_data function in integration.py, it return some fusion fake data.



##  TODO

- Refactor the code
  - n*n combination
  - time offset
  - cross-subjects data
  - EMG action selection
  - pre-processing parameter adjustment
  - make the code configurable
  - testing
- Write some code to build a graph using [MNE-Connectivity](https://mne.tools/mne-connectivity/stable/index.html#) or other methods.
  - try some build_in method
  - native CMC
  - some improved CMC
  - mutual information
  - position-based methods
- Baseline classifier for eeg\emg\eeg-emg data.
  - try EEG-NET
  - try CSP-based classifier
  - try some tree-based methods
- Baseline GNN for graph classification
  - DGL
  - PyG
  - cogdl

