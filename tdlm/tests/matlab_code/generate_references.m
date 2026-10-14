function generate_references()
% Regenerate the MATLAB reference results used by
% tdlm/tests/test_matlab_compatibility.py
%
% run from any directory with:
%   matlab -batch "addpath('tdlm/tests/matlab_code'); generate_references"
% the .mat files are written next to this script.

script_dir = fileparts(mfilename('fullpath'));
old_dir = cd(script_dir);
cleanup = onCleanup(@() cd(old_dir));

% GLM references, each script saves its own .mat file and returns
run_script('Simulate_Replay');
run_script('Simulate_Replay_withAlpha');
run_script('Simulate_Replay_LongerLength');

% cross-correlation reference, lags 0..29 as in the python test
params = load('sequenceness_crosscorr_params.mat');
rd = params.rd;
T = params.T;
lags = 0:29;
sf = nan(1, numel(lags));
sb = nan(1, numel(lags));
for i = 1:numel(lags)
    sf(i) = sequenceness_Crosscorr(rd, T, [], lags(i));
    sb(i) = sequenceness_Crosscorr(rd, T', [], lags(i));
end
save('sequenceness_crosscorr_results.mat', 'sf', 'sb', 'lags');
fprintf('saved sequenceness_crosscorr_results.mat\n');
end


function run_script(name)
% scripts call `clear`, so run them in their own workspace
fprintf('running %s\n', name);
tic;
evalin_script(name);
fprintf('finished %s in %.0f s\n', name, toc);
end


function evalin_script(name)
run(name);
end
