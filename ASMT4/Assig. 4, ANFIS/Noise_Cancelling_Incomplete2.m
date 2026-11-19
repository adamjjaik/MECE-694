%% (1) Create Data for Noise Cancelling Problem
% Load the original sound track
clear
clc

% Load data from MATLAB Library
load NoiseCancelling_Dataset;

% Play the Clean Audio Signal
% sound(Signal_Org, Fs);

% Plot the original sound track
figure(1)
plot(1/Fs:1/Fs:length(Signal_Org)/Fs, Signal_Org, 'DisplayName', 'Sound Track')
title('Information Signal','fontsize',12)
xlabel('Time (seconds)','fontsize',12)
ylabel('Signal','fontsize',12)

% Create white noise and interference data 
NoiseLevel = 0.01;
Interf = NoiseLevel * randn(length(Signal_Org),1) + 2 * Noise;

% Plot the interference data 
figure(1)
hold on
plot(1/Fs:1/Fs:length(Signal_Org)/Fs, Interf, 'DisplayName', 'Interference')
hold off
legend

% The Measured Corrupted Signal
Signal = Signal_Org + Interf;
% Play the Corrupted Audio Signal
% sound(Signal, Fs);

% What we know are: Measured Corrupted Signal and Random Noise (and Not the
% Interference Data)

% GOAL: Recorver Clean Signal "Signal_Org" using "Signal" and "Noise"

%% (2) Create Input/Output Data For ANFIS
% Input of the system is white noise and its delays
x = [Noise [0; Noise(1:end-1)]];

% Output of the system is the Measured Corrupted Signal
y = Signal;

% Divide Data Into Train and Validation Only 
% !!! Write Your Code Here !!!

%% (3) Implement ANFIS
% Implement ANFIS with your choice of settings and your choice of method for
% generating raw FIS
% !!! Write Your Code Here !!!


% Remove Interference from Measured Corrupted Signal
Est_Signal = Signal - Est_Interf;

% Play the Clean Audio Signal
sound(Est_Signal, Fs);

% Plot The Estimated Interference 
figure(1)
hold on
plot(1/Fs:1/Fs:length(Signal_Org)/Fs, Est_Interf, 'DisplayName', 'Estimated Interference')
hold off
legend










