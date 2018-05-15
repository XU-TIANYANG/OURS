% Copy this template configuration file to your VOT workspace.
% Enter the full path to the OURS repository root folder.

OURS_repo_path = '/user/HS228/tx00069/Desktop/CFSDCF_HC';

tracker_label = 'OURS';
tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''OURS'', ''VOT_OURS'', true)', {[OURS_repo_path '/VOT_integration/benchmark_wrapper']});
tracker_interpreter = 'matlab';