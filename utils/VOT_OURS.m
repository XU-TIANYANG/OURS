function results = VOT_OURS(seq,res_path, bSaveImage, parameters)

params = readParams('params.txt');

% Initialize
params.seq = seq;

% Run tracker
results = tracker(params);
