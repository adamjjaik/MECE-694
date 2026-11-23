function [path, cost, nExpanded] = astarGrid(costCube, startIdx, goalIdx, hVec, dirs)
% ASTARGRID  A* shortest path on an 8-connected grid with DIRECTED edge costs.
%
%   [path, cost, nExpanded] = astarGrid(costCube, startIdx, goalIdx, hVec, dirs)
%
%   costCube(r,c,k) : cost of moving from cell (r,c) in direction k (Inf = blocked)
%   startIdx, goalIdx: linear indices of the start and goal cells
%   hVec(n)         : heuristic = lower bound on the remaining cost from node n
%                     (use zeros(N,1) to get Dijkstra's algorithm)
%   dirs            : 8x2 [d_row d_col] direction offsets
%
%   path      : linear indices from start to goal ([] if unreachable)
%   cost      : total cost of the path
%   nExpanded : number of nodes expanded (a measure of search effort)
%
% Uses a binary min-heap written inline: MATLAB has no built-in priority queue,
% and calling a helper per push/pop would copy the heap arrays every time.

    [nR, nC, nD] = size(costCube);
    N       = nR * nC;
    costLin = reshape(costCube, N, nD);
    offs    = dirs(:,1) + dirs(:,2) * nR;      % linear-index offset per direction

    g      = inf(N, 1);  g(startIdx) = 0;      % best known cost from start
    parent = zeros(N, 1);
    closed = false(N, 1);

    cap = 8*N + 16;                            % lazy deletion -> allow duplicates
    hk  = zeros(cap, 1);  hn = zeros(cap, 1);  hs = 1;
    hk(1) = hVec(startIdx);  hn(1) = startIdx;
    nExpanded = 0;

    while hs > 0
        % ---- pop node with smallest f = g + h ----
        u = hn(1);
        hk(1) = hk(hs);  hn(1) = hn(hs);  hs = hs - 1;
        i = 1;
        while true                             % sift down
            l = 2*i;
            if l > hs, break; end
            m = l;
            if l + 1 <= hs && hk(l+1) < hk(l), m = l + 1; end
            if hk(m) < hk(i)
                tk = hk(i); hk(i) = hk(m); hk(m) = tk;
                tn = hn(i); hn(i) = hn(m); hn(m) = tn;
                i = m;
            else
                break
            end
        end

        if closed(u), continue; end            % stale duplicate
        closed(u) = true;
        nExpanded = nExpanded + 1;
        if u == goalIdx, break; end

        % ---- relax the 8 outgoing moves ----
        ru = mod(u - 1, nR) + 1;
        cu = floor((u - 1) / nR) + 1;
        for k = 1:nD
            rv = ru + dirs(k,1);  cv = cu + dirs(k,2);
            if rv < 1 || rv > nR || cv < 1 || cv > nC, continue; end
            v = u + offs(k);
            if closed(v), continue; end
            w = costLin(u, k);
            if ~isfinite(w), continue; end
            gv = g(u) + w;
            if gv < g(v)
                g(v) = gv;  parent(v) = u;
                hs = hs + 1;  hk(hs) = gv + hVec(v);  hn(hs) = v;
                i = hs;
                while i > 1                    % sift up
                    pI = floor(i/2);
                    if hk(pI) > hk(i)
                        tk = hk(i); hk(i) = hk(pI); hk(pI) = tk;
                        tn = hn(i); hn(i) = hn(pI); hn(pI) = tn;
                        i = pI;
                    else
                        break
                    end
                end
            end
        end
    end

    cost = g(goalIdx);
    if ~isfinite(cost), path = []; return; end
    path = goalIdx;
    while path(1) ~= startIdx
        path = [parent(path(1)); path]; %#ok<AGROW>
    end
end
