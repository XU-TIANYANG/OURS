function setup_paths()

[pathstr, name, ext] = fileparts(mfilename('fullpath'));

% Utilities
addpath([pathstr '/utils/']);

addpath([pathstr '/VOT_integration/']);

