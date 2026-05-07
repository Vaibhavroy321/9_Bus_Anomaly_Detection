clc; clear; close all;

%% === PARAMETERS ===
filename = 'steady_state.csv';
Fs = 60; % Sampling frequency (Hz)
T = 1/Fs;

wavelet_level = 3;
wavelet_name = 'db4';
anomaly_threshold_multiplier = 3; % Higher = stricter anomaly detection

buses = 1:9;
fft_signal_name = 'PhX1_0'; % We'll FFT/STFT/Wavelet this waveform

%% === LOAD CSV DATA ===
fid = fopen(filename, 'r');
assert(fid >= 0, 'Cannot open file: %s', filename);

header_line = fgetl(fid);
fclose(fid);

assert(ischar(header_line) || isstring(header_line), 'Empty or invalid header line.');

raw_headers = strsplit(header_line, ';');

valid_names = matlab.lang.makeValidName(raw_headers);
unique_names = matlab.lang.makeUniqueStrings(valid_names);

opts = delimitedTextImportOptions("Delimiter", ";", ...
    "DataLines", 2, ...
    "VariableNames", unique_names, ...
    "VariableTypes", repmat("double", 1, numel(unique_names)));

data_raw = readtable(filename, opts);

%% === Find time column ===
lc_unames = lower(unique_names);
time_guess = find(contains(lc_unames, 'xaxis') | contains(lc_unames, 'time'), 1, 'first');

if ~isempty(time_guess)
    time_col = unique_names{time_guess};
    time = data_raw{:, time_col};
else
    warning('Time column not found. Generating artificial time vector.');
    time = (0:height(data_raw)-1)' * T;
end

L = length(time);
t = time;

%% === Helpers to find columns by signal and bus ===
findColFor = @(sig, bus) localFindColumn(raw_headers, unique_names, sig, bus);

%% === Pre-allocate containers ===
freq_cols = nan(size(buses));
phx_cols = nan(size(buses));

for b = buses
    idxb = find(buses == b);

    c = findColFor('Freq', b);
    if ~isempty(c)
        freq_cols(idxb) = c;
    end

    c = findColFor('PhX1_0', b);
    if ~isempty(c)
        phx_cols(idxb) = c;
    end
end

%% === Analysis functions ===
compute_fft = @(x) localFFT(x, Fs);
[stft_params, stft_fn] = localSTFTParams(64, 32, 128, Fs); %#ok<ASGLU>

%% === Anomaly detection + logging ===
anomaly_logs = {};
wavelet_cd_per_bus = cell(size(buses));
anomaly_idx_per_bus = cell(size(buses));

for b = buses
    idxb = find(buses == b);

    if isnan(phx_cols(idxb)) || phx_cols(idxb) == 0
        wavelet_cd_per_bus{idxb} = [];
        anomaly_idx_per_bus{idxb} = [];
        continue;
    end

    colname = unique_names{phx_cols(idxb)};
    signal_raw = data_raw{:, colname};

    if all(isnan(signal_raw))
        wavelet_cd_per_bus{idxb} = [];
        anomaly_idx_per_bus{idxb} = [];
        continue;
    end

    signal_raw = localFillNaN(signal_raw);

    [C, Lw] = wavedec(signal_raw, wavelet_level + 1, wavelet_name);
    cd = detcoef(C, Lw, wavelet_level);

    wavelet_cd_per_bus{idxb} = cd;

    threshold = anomaly_threshold_multiplier * std(cd);
    anomaly_idx = find(abs(cd) > threshold);

    anomaly_idx_per_bus{idxb} = anomaly_idx;

    if ~isempty(anomaly_idx)
        log_table = table( ...
            repmat({sprintf('PhX1_0_Bus%d', b)}, numel(anomaly_idx), 1), ...
            t(anomaly_idx), ...
            cd(anomaly_idx), ...
            'VariableNames', {'SignalName', 'Time', 'Wavelet_D3_Value'});

        anomaly_logs{end+1} = log_table; %#ok<AGROW>
    end
end

if ~isempty(anomaly_logs)
    writetable(vertcat(anomaly_logs{:}), 'anomaly_log_PhX1_0_Buses1_9.csv');
end

%% === 1) FFT ===
figure('Name', 'FFT - PhX1_0 (Buses 1-9)', 'NumberTitle', 'off');
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for b = buses
    nexttile;
    idxb = find(buses == b);

    if isnan(phx_cols(idxb)) || phx_cols(idxb) == 0
        text(0.5, 0.5, sprintf('PhX1\\_0 Bus %d not found', b), ...
            'HorizontalAlignment', 'center');
        axis off;
        continue;
    end

    x = data_raw{:, unique_names{phx_cols(idxb)}};
    x = localFillNaN(x);

    [fvec, P1] = compute_fft(x);

    plot(fvec, P1);
    grid on;
    title(sprintf('Bus %d', b));
    xlabel('Frequency (Hz)');
    ylabel('|Amplitude|');
end

%% === 2) STFT ===
figure('Name', 'STFT - PhX1_0 (Buses 1-9)', 'NumberTitle', 'off');
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for b = buses
    nexttile;
    idxb = find(buses == b);

    if isnan(phx_cols(idxb)) || phx_cols(idxb) == 0
        text(0.5, 0.5, sprintf('PhX1\\_0 Bus %d not found', b), ...
            'HorizontalAlignment', 'center');
        axis off;
        continue;
    end

    x = data_raw{:, unique_names{phx_cols(idxb)}};
    x = localFillNaN(x);

    [t_stft, fvec, Pmag] = stft_fn(x);

    imagesc(t_stft, fvec, 10*log10(Pmag));
    axis xy;
    xlabel('Time (s)');
    ylabel('Freq (Hz)');
    title(sprintf('Bus %d', b));
    colorbar;
    colormap jet;
end

%% === 3) Wavelet D3 ===
figure('Name', 'Wavelet D3 - PhX1_0 (Buses 1-9)', 'NumberTitle', 'off');
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for b = buses
    nexttile;
    idxb = find(buses == b);

    cd = wavelet_cd_per_bus{idxb};

    if isempty(cd)
        text(0.5, 0.5, sprintf('No data for Bus %d', b), ...
            'HorizontalAlignment', 'center');
        axis off;
        continue;
    end

    plot(cd);
    grid on;
    hold on;

    anomaly_idx = anomaly_idx_per_bus{idxb};

    if ~isempty(anomaly_idx)
        plot(anomaly_idx, cd(anomaly_idx), 'ro', ...
            'MarkerSize', 4, 'LineWidth', 1);
    end

    title(sprintf('Bus %d', b));
    xlabel('Sample Index');
    ylabel('D3 Coeff');
end

%% === 4) Frequency time series ===
figure('Name', 'Frequency (Freq) - Time Series (Buses 1-9)', 'NumberTitle', 'off');
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for b = buses
    nexttile;
    idxb = find(buses == b);

    if isnan(freq_cols(idxb)) || freq_cols(idxb) == 0
        text(0.5, 0.5, sprintf('Freq Bus %d not found', b), ...
            'HorizontalAlignment', 'center');
        axis off;
        continue;
    end

    x = data_raw{:, unique_names{freq_cols(idxb)}};
    x = localFillNaN(x);

    plot(t, x);
    grid on;
    title(sprintf('Bus %d', b));
    xlabel('Time (s)');
    ylabel('Freq (Hz)');
end

%% === 5) PhX1_0 time series ===
figure('Name', 'PhX1_0 - Time Series (Buses 1-9)', 'NumberTitle', 'off');
tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for b = buses
    nexttile;
    idxb = find(buses == b);

    if isnan(phx_cols(idxb)) || phx_cols(idxb) == 0
        text(0.5, 0.5, sprintf('PhX1\\_0 Bus %d not found', b), ...
            'HorizontalAlignment', 'center');
        axis off;
        continue;
    end

    x = data_raw{:, unique_names{phx_cols(idxb)}};
    x = localFillNaN(x);

    plot(t, x);
    grid on;
    title(sprintf('Bus %d', b));
    xlabel('Time (s)');
    ylabel('Signal');
end

disp('Done. Grouped figures created. Anomalies, if any, logged to anomaly_log_PhX1_0_Buses1_9.csv');

%% ===================== Local Functions =====================

function idx = localFindColumn(raw_headers, unique_names, signal, bus)

patterns = { ...
    sprintf('(?i)\\b%s\\b.*\\b%d\\b', regexptranslate('escape', signal), bus), ...
    sprintf('(?i)\\b%d\\b.*\\b%s\\b', bus, regexptranslate('escape', signal)), ...
    sprintf('(?i)%s.*(bus|b)\\s*%d', regexptranslate('escape', signal), bus), ...
    sprintf('(?i)(bus|b)\\s*%d.*%s', bus, regexptranslate('escape', signal)) ...
};

hit = [];

for p = 1:numel(patterns)
    m = ~cellfun('isempty', regexp(raw_headers, patterns{p}, 'once'));

    if any(m)
        hit = find(m, 1, 'first');
        break;
    end
end

if isempty(hit)
    idx = [];
else
    idx = hit;

    if idx < 1 || idx > numel(unique_names)
        idx = [];
    end
end

end

function x = localFillNaN(x)

nan_idx = ~isfinite(x);

if any(nan_idx)
    good = find(~nan_idx);
    bad = find(nan_idx);

    x(nan_idx) = interp1(good, x(good), bad, 'linear', 'extrap');
end

end

function [fvec, P1] = localFFT(x, Fs)

x = x(:);
L = numel(x);

Y = fft(x);
P2 = abs(Y/L);

P1 = P2(1:floor(L/2)+1);

if numel(P1) > 2
    P1(2:end-1) = 2*P1(2:end-1);
end

fvec = Fs * (0:floor(L/2)) / L;

end

function [params, stft_fn] = localSTFTParams(N, noverlap, nfft, Fs)

params = struct('N', N, 'noverlap', noverlap, 'nfft', nfft, 'Fs', Fs);
stft_fn = @(signal_raw) localSTFT(signal_raw, N, noverlap, nfft, Fs);

end

function [t_stft, fvec, Pmag] = localSTFT(signal_raw, N, noverlap, nfft, Fs)

w = hamming(N);
L = numel(signal_raw);

step = N - noverlap;
num_segments = max(0, floor((L - N) / step));

if num_segments == 0
    t_stft = [];
    fvec = linspace(0, Fs/2, nfft/2+1);
    Pmag = zeros(numel(fvec), 1);
    return;
end

spectro_data = zeros(nfft/2+1, num_segments);
t_stft = zeros(1, num_segments);
t = (0:L-1)/Fs;

for k = 1:num_segments
    idx = (k-1)*step + 1;

    segment = signal_raw(idx:idx+N-1) .* w;
    S = fft(segment, nfft);

    spectro_data(:, k) = abs(S(1:nfft/2+1));
    t_stft(k) = t(idx + floor(N/2));
end

fvec = linspace(0, Fs/2, nfft/2+1);
Pmag = spectro_data;

end