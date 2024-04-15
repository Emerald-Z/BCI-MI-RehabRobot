# Structure of the data

Note: 4 and 6 in this case are simply based on the number of runs in the online sessions.
This will change between subjects.

## Diagram

SubjectN
│
├─offline                           - Data from offline session
│ ├─rest                            - Contains rest periods
│ │ ├─{cell array: 2x1}             - Each cell holds EEG data for a rest period
│ │
│ └─runs                            - Contains run data split into task periods
│   ├─eeg                           - EEG data from task runs
│   │ ├─{cell array: 3x1}           - Each cell holds EEG data for one run, sub-divided into trials
│   │
│   └─labels                        - Labels for the task runs
│     ├─{cell array: 3x1}           - Each cell holds labels indicating task type (rest or reach) for each trial
│
└─online                            - Data from online sessions
  ├─session2                        - Data from the first online session
  │ ├─eeg                           - EEG data from task runs
  │ │ ├─{cell array: 4x1}           - Each cell holds EEG data for one run, sub-divided into trials
  │ │
  │ └─labels                        - Labels for the task runs, including type, end, and sustain
  │   ├─type                        - Task type labels (rest or reach)
  │   ├─end                         - Indicates if initial command delivery was correct
  │   └─sustain                     - Indicates if the task was sustained successfully
  │
  └─session3                        - Data from the second online session
    ├─rest                          - Contains rest periods
    │ ├─{cell array: 2x1}           - Each cell holds EEG data for a rest period
    │
    ├─eeg                           - EEG data from task runs
    │ ├─{cell array: 6x1}           - Each cell holds EEG data for one run, sub-divided into trials
    │
    └─labels                        - Labels for the task runs, including type, end, and sustain
      ├─type                        - Task type labels (rest or reach)
      ├─end                         - Indicates if initial command delivery was correct
      └─sustain                     - Indicates if the task was sustained successfully
