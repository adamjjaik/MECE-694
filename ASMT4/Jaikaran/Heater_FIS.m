%% Heater_FIS.m
% MECE 694 - Assignment 4, Section 1: Mamdani fuzzy heater controller
% Inputs : TE  = Temperature Error = T_desired - T_current  [-10, 10] degC
%          RTC = Rate of Temperature Change                  [-5, 5]  degC/min
% Output : HC  = Heater Control                              [0, 100] %
% Design : 3 MFs per variable, triangle in the middle, trapezoids at the ends
%          3 x 3 = 9 rules
% Based on the structure of AutoDriving_FISController.m (course sample code)
clc; clear; close all;

%% (1) FIS structure - Mamdani
FIS = mamfis("Name", "HeaterControl", ...
    "AndMethod", "min", ...              % t-norm for AND
    "OrMethod", "max", ...               % s-norm for OR
    "ImplicationMethod", "min", ...      % clip output MF at rule strength
    "AggregationMethod", "max", ...      % combine clipped output MFs
    "DefuzzificationMethod", "centroid");% crisp HC = centre of area

%% (2) Inputs, output and membership functions
% Adjacent MFs cross at membership 0.5 -> every input value is covered.

% --- Input 1: Temperature Error (positive = room colder than setpoint)
FIS = addInput(FIS, [-10 10], "Name", "TE");
FIS = addMF(FIS, "TE", "trapmf", [-10 -10 -4 0], "Name", "Warm");     % room above setpoint
FIS = addMF(FIS, "TE", "trimf",  [-4 0 4],       "Name", "OnTarget"); % near setpoint
FIS = addMF(FIS, "TE", "trapmf", [0 4 10 10],    "Name", "Cold");     % room below setpoint

% --- Input 2: Rate of Temperature Change
FIS = addInput(FIS, [-5 5], "Name", "RTC");
FIS = addMF(FIS, "RTC", "trapmf", [-5 -5 -1.5 0], "Name", "Falling");
FIS = addMF(FIS, "RTC", "trimf",  [-1.5 0 1.5],   "Name", "Steady");
FIS = addMF(FIS, "RTC", "trapmf", [0 1.5 5 5],    "Name", "Rising");

% --- Output: Heater Control (%)
FIS = addOutput(FIS, [0 100], "Name", "HC");
FIS = addMF(FIS, "HC", "trapmf", [0 0 20 50],    "Name", "Low");
FIS = addMF(FIS, "HC", "trimf",  [20 50 80],     "Name", "Medium");
FIS = addMF(FIS, "HC", "trapmf", [50 80 100 100], "Name", "High");

%% (3) Rule base: 3 x 3 = 9 rules
% Columns: [TE_MF  RTC_MF  HC_MF  weight  connection(1 = AND)]
% TE : 1 Warm, 2 OnTarget, 3 Cold
% RTC: 1 Falling, 2 Steady, 3 Rising
% HC : 1 Low, 2 Medium, 3 High
%
%                 RTC:  Falling   Steady   Rising
%   TE = Warm           Low       Low      Low
%   TE = OnTarget       Medium    Low      Low
%   TE = Cold           High      High     Medium
Rules = [
    1 1 1 1 1
    1 2 1 1 1
    1 3 1 1 1
    2 1 2 1 1
    2 2 1 1 1
    2 3 1 1 1
    3 1 3 1 1
    3 2 3 1 1
    3 3 2 1 1 ];
FIS = addRule(FIS, Rules);

% Save so it can be opened in Fuzzy Logic Designer for report screenshots:
%   fuzzyLogicDesigner('HeaterControl.fis')
writeFIS(FIS, "HeaterControl");

%% (4) Architecture plots for the report
figure('Name', 'FIS architecture');  plotfis(FIS);

figure('Name', 'Membership functions');
subplot(3,1,1), plotmf(FIS, 'input', 1),  title('Temperature Error (TE)'),          xlabel('TE (\circC)')
subplot(3,1,2), plotmf(FIS, 'input', 2),  title('Rate of Temperature Change (RTC)'), xlabel('RTC (\circC/min)')
subplot(3,1,3), plotmf(FIS, 'output', 1), title('Heater Control (HC)'),             xlabel('HC (%)')

figure('Name', 'Control surface');
gensurf(FIS);
xlabel('TE (\circC)'), ylabel('RTC (\circC/min)'), zlabel('HC (%)')
title('Heater control surface')

%% (5) Simulate test scenarios
Scenarios = [
%    TE    RTC
     8    -2     % very cold, still cooling        -> expect High (max output)
     8     3     % very cold, warming fast          -> expect Medium-High (eases off)
     3     0     % slightly cold, steady            -> expect between Medium and High
     0     0     % at setpoint, steady              -> expect Low (cover heat loss)
     0    -2     % at setpoint, cooling             -> expect Medium
     0     2     % at setpoint, warming             -> expect Low
    -3    -1     % slightly warm, cooling           -> expect Low to Medium
    -8     0     % much too warm                    -> expect Low (minimum output)
    10    -5     % extreme cold, extreme cooling    -> expect max output
    ];

ev = evalfisOptions('OutOfRangeInputValueMessage', 'none', 'NoRuleFiredMessage', 'warning');
HC = evalfis(FIS, Scenarios, ev);

ScenarioTable = table(Scenarios(:,1), Scenarios(:,2), round(HC, 1), ...
    'VariableNames', {'TE_degC', 'RTC_degC_per_min', 'HC_percent'});
disp(ScenarioTable)

% Rule viewer for one scenario (screenshot for the report)
figure('Name', 'Rule viewer');
plotrule(FIS, 'Inputs', [3 0]);

%% (6) HC vs TE at three rates of change
TEv = linspace(-10, 10, 201)';
figure('Name', 'HC vs TE');
hold on
for r = [-2 0 2]
    plot(TEv, evalfis(FIS, [TEv, r*ones(size(TEv))], ev), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('RTC = %g \\circC/min', r));
end
hold off, grid on, legend('Location', 'northwest')
xlabel('TE (\circC)'), ylabel('HC (%)'), title('Heater output vs temperature error')