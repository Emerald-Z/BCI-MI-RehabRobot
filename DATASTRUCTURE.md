# Data Structure

Note: Dims 4 and 6 are simply due to the number of runs for subject 1 (Nihita). Edit these per subject

### Offline Data
- `offline`
  - `rest`
    - `cell array (2x1)`
      - Contains EEG data for rest periods.
  - `runs`
    - `eeg`
      - `cell array (3x1)`
        - Contains EEG data for each run, sub-divided into trials.
    - `labels`
      - `cell array (3x1)`
        - Contains labels for task type (rest or reach) for each trial.

### Online Data
- `online`
  - `session2`
    - `eeg`
      - `cell array (4x1)`
        - Contains EEG data for each run, sub-divided into trials.
    - `labels`
      - `type`
        - Task type labels (rest or reach).
      - `end`
        - Indicates if initial command delivery was correct.
      - `sustain`
        - Indicates if the task was sustained successfully.
  - `session3`
    - `rest`
      - `cell array (2x1)`
        - Contains EEG data for rest periods.
    - `eeg`
      - `cell array (6x1)`
        - Contains EEG data for each run, sub-divided into trials.
    - `labels`
      - `type`
        - Task type labels (rest or reach).
      - `end`
        - Indicates if initial command delivery was correct.
      - `sustain`
        - Indicates if the task was sustained successfully.
