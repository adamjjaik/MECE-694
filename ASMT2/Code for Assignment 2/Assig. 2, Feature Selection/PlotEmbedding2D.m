function PlotEmbedding2D(Z, y, ActivityNames, TitleStr, ShowLegend)
% PlotEmbedding2D  2D scatter of an embedding, coloured by activity label.
%   Z: n x >=2 embedding, y: n x 1 labels (1..6). Labels are used for colour only.
Colors = lines(6);                       % fixed colour per activity across all figures
hold on
for c = 1:6
    m = (y == c);
    scatter(Z(m,1), Z(m,2), 6, Colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6, ...
            'DisplayName', strrep(ActivityNames{c}, '_', ' '));
end
hold off; grid on; box on
xlabel('Dimension 1'); ylabel('Dimension 2'); title(TitleStr);
if nargin > 4 && ShowLegend, legend('Location', 'bestoutside'); end
end