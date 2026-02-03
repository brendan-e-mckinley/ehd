% ------------------------------------------------------
% Load geometry
% ------------------------------------------------------
ld = load('Windmill_Geom.mat');

xib = ld.xib;
yib = ld.yib;

scatter(xib, yib, 40, 'k', 'filled')

% Interpolate along open curve
pt = interparc(8000, xib, yib, 'spline');

x = pt(:,1);
y = pt(:,2);

x(end) = [];
y(end) = [];
N = length(x);

% --- Subsample boundary ---
num_pts = 81;
idx = round(linspace(1, N, num_pts));
x = x(idx);
y = y(idx);
x(1) = [];
y(1) = [];
N = num_pts;

% ------------------------------------------------------
% Compute unit tangents
% ------------------------------------------------------
dx = zeros(N,1);
dy = zeros(N,1);

% Central differences
dx(2:N-1) = (x(3:N) - x(1:N-2)) / 2;
dy(2:N-1) = (y(3:N) - y(1:N-2)) / 2;

% One-sided at endpoints
dx(1)   = x(2)   - x(1);
dy(1)   = y(2)   - y(1);
dx(end) = x(end) - x(end-1);
dy(end) = y(end) - y(end-1);

% Normalize
tmag = sqrt(dx.^2 + dy.^2);
tx = dx ./ tmag;
ty = dy ./ tmag;

% ------------------------------------------------------
% Unit normal vectors (components on unit circle)
% ------------------------------------------------------
nx = -ty;
ny = tx;

% Sanity check: ||n|| = 1
% max(abs(sqrt(nx.^2 + ny.^2) - 1))

% ------------------------------------------------------
% Visualization (optional)
% ------------------------------------------------------
figure
scatter(x, y, 40, 'k', 'filled')
hold on
quiver(x, y, nx, ny, 0.25, 'r', 'LineWidth', 1.2)
axis equal
grid on
title('Boundary Points with Unit Normals')
xlabel('x')
ylabel('y')
legend('Boundary points', 'Unit normals')


save('Windmill_Geom_Larger_Next_Finer.mat', 'x', 'y', 'nx', 'ny')

