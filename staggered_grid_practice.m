%% staggered_poisson_fixed.m
% Staggered-grid solver, consistent flattening and U-ghost (-3) applied only to U
clear; clc; close all;

%% Parameters
Nx = 64;           % interior points in x (pressure/U grid)
Ny = Nx;           % interior points in y
L = 1;
x = linspace(0, L, Nx+2);   % includes boundary nodes
dx = x(2) - x(1); dx2 = dx^2;
y = x; dy = dx;

% interior coordinates (pressure / primary grid)
xint = x(2:end-1);    % length Nx
yint = y(2:end-1);    % length Ny

% staggered coordinates
x_i_U = xint - dx/2;          % U is staggered in x
y_j_V = yint - dy/2;          % V is staggered in y
y_j_V(end) = [];              % Ny-1 values

% counts
N_U = Nx * Ny;
Ny_minus = Ny - 1;
N_V = Nx * Ny_minus;
N_P = Nx * Ny;    % pressure DOFs

%% Physical BCs
V_bottom = -3.5;
V_top    = -3.5;

f_bc_mat = zeros(Nx, Ny);         % U RHS (cols = y-levels)
g_bc_mat = zeros(Nx, Ny_minus);   % V RHS
h_bc_mat = zeros(Nx, Ny);         % pressure RHS

%% Sigma
compute_sigma = @(xx, yy) 50 * (sin(2*pi*xx).*sin(4*pi*yy) + 1);

sigma_matrix_U = zeros(Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        sigma_matrix_U(i,j) = compute_sigma(x_i_U(i), yint(j));
    end
end
sigma_U_diag = spdiags(sigma_matrix_U(:), 0, N_U, N_U);

sigma_matrix_V = zeros(Nx, Ny_minus);
for i = 1:Nx
    for j = 1:Ny_minus
        sigma_matrix_V(i,j) = compute_sigma(xint(i), y_j_V(j));
    end
end
sigma_V_diag = spdiags(sigma_matrix_V(:), 0, N_V, N_V);

%% 1-D operators
% x-direction (periodic)
D2_x = spdiags([ones(Nx,1), -2*ones(Nx,1), ones(Nx,1)], [-1 0 1], Nx, Nx);
D2_x(1,end) = 1; 
D2_x(end,1) = 1;
D2_x = D2_x / dx2;

% y-direction for U (includes ghost treatment -> -3 on boundaries)
D2_y_U = spdiags([ones(Ny,1), -2*ones(Ny,1), ones(Ny,1)], [-1 0 1], Ny, Ny);
D2_y_U(1,1) = -3;
D2_y_U(end,end) = -3;
D2_y_U = D2_y_U / dy^2;

% y-direction for V (interior Ny-1 points)
D2_y_V = spdiags([ones(Ny_minus,1), -2*ones(Ny_minus,1), ones(Ny_minus,1)], [-1 0 1], Ny_minus, Ny_minus) / dy^2;

%% Laplacians (consistent ordering: x varies fastest; use kron(speye(Ny), D2_x) + kron(D2_y, speye(Nx)))
Lap_U = kron(speye(Ny), D2_x) + kron(D2_y_U, speye(Nx));    % N_U x N_U
Lap_V = kron(speye(Ny_minus), D2_x) + kron(D2_y_V, speye(Nx)); % N_V x N_V

%% Gradients (canonical ordering with x fastest)
% G_x: forward difference in x with periodic wrap (Nx x Nx)
D_x_forward = spdiags([-ones(Nx,1), ones(Nx,1)], [0 1], Nx, Nx);
D_x_forward(end,1) = 1;
D_x_forward = D_x_forward / dx;

% G_x maps P (N_P) -> U (N_U)
G_x = kron(speye(Ny), D_x_forward);   % N_U x N_P

% G_y: vertical diff from P (Ny) to V (Ny-1) per column
D_y_small = spdiags([-ones(Ny_minus,1), ones(Ny_minus,1)], [0 1], Ny_minus, Ny);
D_y_small = D_y_small / dy;

G_y = kron(D_y_small, speye(Nx));     % N_V x N_P

% Divergence (transpose of gradient with chosen sign)
D_x = -G_x.';  % N_P x N_U
D_y = -G_y.';  % N_P x N_V

%% Automatic BC contributions (consistent with the above sign conv)
% V Dirichlet at top/bottom produces source terms in V-Laplacian (g_bc)
% and in pressure divergence (h_bc). We'll compute them in matrix form then vectorize.

% g_bc: effect of substituting V_boundary into the 1D D2 stencil for the first and last interior row
% recall g_bc_mat shape is Nx x (Ny-1) where columns index interior V rows (j = 1..Ny-1)
g_bc_mat = zeros(Nx, Ny_minus);
% bottom boundary contributes to first interior V row
g_bc_mat(:,1) = V_bottom / dy^2;
% top boundary contributes to last interior V row
g_bc_mat(:,end) = V_top / dy^2;
g_bc = g_bc_mat(:);

% h_bc: the pressure equation contains D_y * V; boundary V enters as -V_boundary/dy in pressure RHS
% with our D_y = -G_y.' convention the sign is:
% For bottom pressure row: contribution = - V_bottom / dy
% For top pressure row: contribution = + V_top / dy  (depends on indexing orientation)
% We'll derive by applying D_y to a V vector that has zero interior and V_boundary nonzero,
% but easier: fill h_bc_mat entries that correspond to the bottom and top pressure rows.
h_bc_mat = zeros(Nx, Ny);
% bottom pressure row (j=1): D_y includes a -1/dy * V_bottom term => contribution = -V_bottom/dy
h_bc_mat(:,1) = - V_bottom / dy;
% top pressure row (j=Ny): D_y includes a +1/dy * V_top term when D_y = -G_y.' so net is +V_top/dy
h_bc_mat(:,end) =  V_top / dy;
h_bc = h_bc_mat(:);

%% Build saddle point matrix
A11 = Lap_U + sigma_U_diag;   A12 = sparse(N_U, N_V);  A13 = G_x;
A21 = sparse(N_V, N_U);       A22 = Lap_V + sigma_V_diag; A23 = G_y;
A31 = D_x;                    A32 = D_y;                A33 = sparse(N_P, N_P);

A = [A11, A12, A13;
     A21, A22, A23;
     A31, A32, A33];

% Stack RHS: interior sources are zero; BC contributions included via g_bc/h_bc
RHS = [ zeros(N_U,1);    % f = 0 interior
        g_bc;            % from Dirichlet V BCs entering V-Laplacian
        h_bc ];          % from Dirichlet V BCs entering divergence eqn

%% Adjoint consistency test (should be ~machine eps)
p = randn(N_P,1); u = randn(N_U,1); v = randn(N_V,1);
lhs = (G_x*p)'*u + (G_y*p)'*v;
rhs = - p'*(D_x*u + D_y*v);
fprintf('Adjoint mismatch (should be tiny): %g\n', abs(lhs-rhs));

%% Solve
fprintf('Assembling A: size %d x %d\n', size(A,1), size(A,2));
sol = A \ RHS;

% split
U = sol(1:N_U);
V = sol(N_U+1:N_U+N_V);
P = sol(N_U+N_V+1:end);

%% Residual and diagnostics
r = A*sol - RHS;
fprintf('Residual norm: %g, max abs residual: %g\n', norm(r), max(abs(r)));
[~, idx_max] = max(abs(r));
if idx_max <= N_U
  fprintf('largest residual in U block, local index %d\n', idx_max);
elseif idx_max <= N_U+N_V
  local = idx_max - N_U;
  fprintf('largest residual in V block, local index %d\n', local);
else
  local = idx_max - N_U - N_V;
  fprintf('largest residual in P block, local index %d\n', local);
end

%% Reshape for plotting (matching flattening: reshape(mat,[Nx,Ny]) for x fastest)
Uplot = reshape(U, [Nx, Ny]);           % Nx x Ny: rows=x, cols=y
Vplot = reshape(V, [Nx, Ny_minus]);     % Nx x (Ny-1)
Pplot = reshape(P, [Nx, Ny]);           % Nx x Ny

% Build full arrays for plotting convenience
UFull = NaN(Nx+2, Ny+2);
UFull(2:Nx+1, 2:Ny+1) = Uplot;
UFull(:,1) = UFull(:,Nx+1);   % periodic left ghost (x-direction)
UFull(:,Nx+2) = UFull(:,2);   % periodic right ghost

VFull = V_bottom * ones(Nx+2, Ny+2);     % initialize with Dirichlet V
VFull(2:Nx+1, 2:Ny) = -Vplot;             % interior V
VFull(:,1) = VFull(:,Nx+1); VFull(:,Nx+2) = VFull(:,2);

% Plot (meshgrid with x rows, y cols)
xplot = linspace(0,L,Nx+2);
yplot = linspace(0,L,Ny+2);
[Xplot, Yplot] = meshgrid(yplot, xplot);

figure; surf(Xplot, Yplot, UFull, 'EdgeColor','none'); title('U'); xlabel('y'); ylabel('x');
figure; surf(Xplot, Yplot, VFull, 'EdgeColor','none'); title('V'); xlabel('y'); ylabel('x');
figure; surf(Xplot, Yplot, sqrt(UFull.^2 + VFull.^2), 'EdgeColor','none'); title('|velocity|');

