function PlotEmbedding3D(Z, y, ActivityNames, TitleStr, ShowLegend)
% PlotEmbedding3D  3D scatter of an embedding, coloured by activity (colour only).
Colors = lines(6);
hold on
for c = 1:6
    m = (y == c);
    scatter3(Z(m,1), Z(m,2), Z(m,3), 6, Colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6, ...
             'DisplayName', strrep(ActivityNames{c}, '_', ' '));
end
hold off; grid on; box on; view(45, 25)
xlabel('Dimension 1'); ylabel('Dimension 2'); zlabel('Dimension 3'); title(TitleStr);
if nargin > 4 && ShowLegend, legend('Location', 'bestoutside'); end
end