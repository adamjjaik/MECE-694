function PlotPredictions(FIS, Data, delays, Label)
% Plot true vs predicted series for train and test data, ordered in time.
    ev = evalfisOptions('OutOfRangeInputValueMessage', 'none', ...
                        'NoRuleFiredMessage', 'none');
    Sets = {'train', 'test'};
    figure('Name', char(Label));
    for s = 1:2
        X   = Data.(['X_' Sets{s}]);
        Y   = Data.(['Y_' Sets{s}]);
        idx = Data.([Sets{s} 'Idx']);

        t = idx(:) + max(delays);          % row number -> time index of x
        [t, order] = sort(t);              % random split -> re-order in time
        Yhat = evalfis(FIS, X(order, :), ev);

        subplot(2, 1, s)
        plot(t, Y(order), 'b-', 'LineWidth', 1, 'DisplayName', 'True'); hold on
        plot(t, Yhat, 'r.', 'MarkerSize', 8, 'DisplayName', 'FIS prediction'); hold off
        xlabel('Time index t'), ylabel('x(t)'), legend('Location', 'best'), grid on
        title(sprintf('%s data - %s', Sets{s}, Label), 'Interpreter', 'none')
    end