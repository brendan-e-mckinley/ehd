%% staggered_poisson_script.m
% Full script with corrected operators for staggered grid
% - ordering: reshape(vec, [Nx, Ny]) => x index varies fastest (row index)
% - Laplacians use consistent kron ordering: kron(speye(Ny), Dx) + kron(Dy, speye(Nx))
% - Gradients and divergence consistent with that ordering

clear; clc; close all;

%% Setup
Nx = 3;            % number of grid points in x-direction (physical P/U grid)
Ny = Nx;            % number of grid points in y-direction (physical P/U grid)
L = 1;
x = linspace(0, L, Nx+2);   % include ghost/boundary nodes (Nx interior + 2)
dx = x(2) - x(1);
dx2 = dx * dx;
y = x;
dy = y(2) - y(1);

% interior (pressure/primary) coordinates (exclude the first/last which are boundaries)
xint = x(2:end-1);    % length Nx
yint = y(2:end-1);    % length Ny

[Xint, Yint] = meshgrid(xint, yint); % Ny x Nx (note meshgrid: rows=y, cols=x)

% Staggered grids
% U lives at x_i_U = xint - dx/2, y = yint   -> size Nx x Ny (same count as P)
x_i_U = xint - dx/2;
% V lives at x = xint, y_j_V = yint - dy/2   -> size Nx x (Ny-1)
y_j_V = yint - dy/2;
y_j_V(end) = [];  % now length Ny-1

% Degrees of freedom counts
N_U = Nx * Ny;
Ny_minus = Ny - 1;
N_V = Nx * Ny_minus;
N_P = Nx * Ny;   % pressure same as U grid count

%% RHS (use matrix shapes consistent with reshape ordering)
% We will represent rhs arrays with rows corresponding to x (1..Nx) and columns to y (1..Ny)
% That matches reshape(vec,[Nx,Ny]) below where x varies fastest.

f_bc_mat = zeros(Nx, Ny);           % U RHS (Nx x Ny)
g_bc_mat = zeros(Nx, Ny_minus);     % V RHS (Nx x (Ny-1))
h_bc_mat = zeros(Nx, Ny);           % Pressure RHS (Nx x Ny)

% Boundary contributions for V and pressure due to Dirichlet V = -3.5 at y boundaries.
% These follow the pattern in your original code but transposed to Nx-by-Ny shapes.
g_bc_mat(:,1)        =  3.5 / dx2;   % lower V-row effect (adjacent to bottom BC)
g_bc_mat(:,Ny_minus) =  3.5 / dx2;   % upper V-row effect (adjacent to top BC)

% pressure RHS contributions due to vertical BC (as in your original code):
h_bc_mat(:,1) =  3.5 / dx;    % bottom boundary contribution
h_bc_mat(:,end) = -3.5 / dx;  % top boundary contribution

% Convert to column vectors consistent with operator ordering (x fastest)
f_bc = f_bc_mat(:);
g_bc = g_bc_mat(:);
h_bc = h_bc_mat(:);

%% Sigma
compute_sigma = @(xx, yy) 50 * (sin(2 * pi * xx) .* sin(4 * pi * yy) + 1);

% sigma for U grid (Nx x Ny) at (x_i_U, yint)
sigma_matrix_U = zeros(Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        sigma_matrix_U(i, j) = compute_sigma(x_i_U(i), yint(j));
    end
end
sigma_U_diag = spdiags(sigma_matrix_U(:), 0, N_U, N_U);

% sigma for V grid (Nx x (Ny-1)) at (xint, y_j_V)
sigma_matrix_V = zeros(Nx, Ny_minus);
for i = 1:Nx
    for j = 1:Ny_minus
        sigma_matrix_V(i, j) = compute_sigma(xint(i), y_j_V(j));
    end
end
sigma_V_diag = spdiags(sigma_matrix_V(:), 0, N_V, N_V);

%% Laplacian operators
% 1D second-derivative in x (periodic)
main_diag = -2 * ones(Nx, 1);
off_diag  = ones(Nx, 1);
D2_x = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], Nx, Nx);
D2_x(1, end) = 1;
D2_x(end, 1) = 1;
D2_x = D2_x / dx2;

% 1D second-derivative in y for P/U (Dirichlet in y direction)
main_diag_y = -2 * ones(Ny, 1);
off_diag_y  = ones(Ny, 1);
D2_y = spdiags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], Ny, Ny);
D2_y(1,1) = -3;
D2_y(end,end) = -3;
% Note: For interior-only formulation you may normally modify corners or account BC in RHS.
D2_y = D2_y / dx2;

spy(D2_y);

% Build Lap_U: size N_U x N_U
% Use ordering: x varies fastest => Lap_U = kron(speye(Ny), D2_x) + kron(D2_y, speye(Nx))
Lap_U = kron(speye(Ny), D2_x) + kron(D2_y, speye(Nx));
% Lap_U is Nx*Ny by Nx*Ny
spy(Lap_U);
title('Lap\_U sparsity');

% Laplacian for V: x periodic (Nx), y has Ny_minus points (internal V rows)
main_diag_yv = -2 * ones(Ny_minus, 1);
off_diag_yv  = ones(Ny_minus, 1);
D2_y_V = spdiags([off_diag_yv, main_diag_yv, off_diag_yv], [-1, 0, 1], Ny_minus, Ny_minus);
D2_y_V = D2_y_V / dx2;

% Lap_V: size N_V x N_V
Lap_V = kron(speye(Ny_minus), D2_x) + kron(D2_y_V, speye(Nx));
spy(Lap_V);
title('Lap\_V sparsity');

%% Gradients (consistent with ordering)
% Gradient in x: forward difference with periodic wrap, size Nx x Nx
D_x_forward = spdiags([-ones(Nx,1), ones(Nx,1)], [0,1], Nx, Nx);
D_x_forward(end,1) = 1;
D_x_forward = D_x_forward / dx;

% G_x maps P (N_P) -> U (N_U) with size (N_U x N_P)
G_x = kron(speye(Ny), D_x_forward);
spy(G_x);
title('G\_x sparsity');

% Gradient in y: maps P (Nx*Ny) -> V (Nx*(Ny-1))
% For each x-column: (p(j+1) - p(j)) / dy for j = 1..Ny-1
D_y_small = spdiags([-ones(Ny_minus,1), ones(Ny_minus,1)], [0,1], Ny_minus, Ny);
D_y_small = D_y_small / dy;

% G_y size: N_V x N_P
G_y = kron(D_y_small, speye(Nx));
spy(G_y);
title('G\_y sparsity');

%% Divergence operators (transpose of gradients), choose sign convention consistently
% Here choose D = -G' so that D*Vel + ... is consistent with gradient signs used.
D_x = -G_x.';
D_y = -G_y.';

spy(D_y);
title('D\_y sparsity');

%% Assemble saddle-point matrix
% Blocks:
% [ Lap_U + sigma_U,   0,        G_x;
%   0,                 Lap_V + sigma_V,  G_y;
%   D_x,               D_y,      0 ]

A11 = Lap_U + sigma_U_diag;            % N_U x N_U
A12 = sparse(N_U, N_V);
A13 = G_x;                             % N_U x N_P

A21 = sparse(N_V, N_U);
A22 = Lap_V + sigma_V_diag;            % N_V x N_V
A23 = G_y;                             % N_V x N_P

A31 = D_x;                             % N_P x N_U
A32 = D_y;                             % N_P x N_V
A33 = sparse(N_P, N_P);

% Assemble A (sparse)
A = [A11, A12, A13;
     A21, A22, A23;
     A31, A32, A33];

% Build RHS (stacked)
RHS = [ f_bc;
        g_bc;
        h_bc; ];

%% Sanity checks
fprintf('Sizes: N_U=%d, N_V=%d, N_P=%d\n', N_U, N_V, N_P);
fprintf('A size: %d x %d\n', size(A,1), size(A,2));
% quick size asserts
assert(size(A,1) == (N_U + N_V + N_P), 'A size mismatch');

%% Solve (direct)
fprintf('Solving linear system (direct)...\n');
sol = A \ RHS;

% split solution
U = sol(1:N_U);
V = sol(N_U + (1:N_V));
P = sol(N_U + N_V + (1:N_P));

% Residual check
r = A * sol - RHS;
fprintf('Residual norm: %g, max abs residual: %g\n', norm(r), max(abs(r)));

%% Reshape for plotting (remember ordering: reshape(vec, [Nx, Ny]) => x varies fastest)
Uplot = reshape(U, [Nx, Ny]);           % matrix Nx x Ny
Vplot = reshape(V, [Nx, Ny_minus]);    % matrix Nx x (Ny-1)
Pplot = reshape(P, [Nx, Ny]);           % matrix Nx x Ny

% Build full arrays including boundary rows/cols for plotting convenience
% Create a grid including ghost/boundary nodes (Nx+2 x Ny+2) like your original script
X = repmat(x', 1, Ny+2);  % (Nx+2) x (Ny+2) but we only need for surf demonstration
Y = repmat(y, Nx+2, 1);

UFull = NaN(Nx+2, Ny+2);
UFull(2:Nx+1, 2:Ny+1) = Uplot;   % place interior U values
% periodic in x: copy column 2 into column 1 (left ghost) and column Ny+1 into column Ny+2 (right ghost)
UFull(1, :) = UFull(Nx+1, :);    % left periodic ghost (approx)
UFull(Nx+2, :) = UFull(2, :);    % right ghost (approx)

VFull = -3.5 * ones(Nx+2, Ny+2);  % initialize with Dirichlet V = -3.5
VFull(2:Nx+1, 2:Ny) = -Vplot;      % interior V occupies columns 2..Ny (since Ny_minus = Ny-1)
VFull(1, :) = VFull(Nx+1, :);
VFull(Nx+2, :) = VFull(2, :);

PFull = NaN(Nx+2, Ny+2);
PFull(2:Nx+1, 2:Ny+1) = Pplot;
PFull(1, :) = PFull(Nx+1, :);
PFull(Nx+2, :) = PFull(2, :);

% Create mesh for plotting (interior xint,yint + ghost)
xplot = linspace(0, L, Nx+2);
yplot = linspace(0, L, Ny+2);
[Xplot, Yplot] = meshgrid(yplot, xplot);  % note: meshgrid(y,x) to align with rows= x

figure;
surf(Xplot, Yplot, UFull, 'EdgeColor', 'none');
title('U (including ghost columns)');
xlabel('y'); ylabel('x'); zlabel('U');

figure;
surf(Xplot, Yplot, VFull, 'EdgeColor', 'none');
title('V (including BCs)');
xlabel('y'); ylabel('x'); zlabel('V');

figure;
Umean = sqrt( (UFull.^2) + (VFull.^2) );
surf(Xplot, Yplot, Umean, 'EdgeColor', 'none');
title('Speed magnitude (approx)');
xlabel('y'); ylabel('x'); zlabel('|U,V|');

%% Quick diagnostics: where residual is largest
[~, idx_max] = max(abs(r));
fprintf('Index of largest residual: %d (row), value = %g\n', idx_max, r(idx_max));

% End of script
