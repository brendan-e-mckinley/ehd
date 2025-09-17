clear;clc

%% Setup
Nx = 64; % number of grid points along one direction
Ny = Nx;
L = 1; 
x = linspace(0,L,Nx+2);
dx = x(2)-x(1);
dx2 = dx * dx;
y = x;
dy = y(2)-y(1);

xint = x(2:end-1);
yint = y(2:end-1);
%xint = x;
%yint = y;

[Xint,Yint] = meshgrid(xint,yint); % make 2D grid of interior points
[X,Y] = meshgrid(x, y);

N_U = Nx*Ny;
N_V = Nx*(Ny+1);

x_i_U = xint - dx/2;

y_j_V = yint - dy/2;
y_j_V(end+1) = yint(end) + dy/2;

%% RHS

f_bc = zeros([Ny Nx]);
g_bc = zeros([Ny+1, Nx]);
h_bc = zeros([Ny Nx]);

g_bc(1,:) = 3.5/dx2;
g_bc(Ny+1,:) = 3.5/dx2;

h_bc(1,:) = 3.5/dx;
h_bc(Ny,:) = -3.5/dx;
figure(1);
spy(g_bc(:));

%% Sigma

compute_sigma = @(x, y) 50 * (sin(2 * pi * x) * sin(4 * pi * y) + 1);

sigma_matrix_U = zeros([Nx Ny]);
for k = 1:Nx
    for j = 1:Ny
        sigma_matrix_U(k, j) = compute_sigma(x_i_U(k), yint(j));
    end
end

sigma_U_diag = diag(sigma_matrix_U(:));

sigma_matrix_V = zeros([Nx Ny+1]);
for k = 1:Nx
    for j = 1:Ny+1
        sigma_matrix_V(k, j) = compute_sigma(xint(k), y_j_V(j));
    end
end

sigma_V_diag = diag(sigma_matrix_V(:));

%% Laplacian

% Lap_U = 1/dx2 * laplacian_2d_kron(Nx, Ny);
% Lap_V = laplacian_2d_kron(Nx-1, Ny);
% 
% spy(Lap_V)
% 
% function A = laplacian_2d_kron(m, n)
%     % 1D operators
%     D1_y = dirichlet_laplacian_1d(m);  % m x m (Dirichlet in y-direction)
%     D1_x = periodic_laplacian_1d(n);   % n x n (periodic in x-direction)
% 
%     I_m = speye(m);
%     I_n = speye(n);
% 
%     % - I_m ⊗ D1_x handles horizontal connections (within rows)
%     % - D1_y ⊗ I_n handles vertical connections (between rows)
%     A = kron(I_m, D1_x) + kron(D1_y, I_n);
% end
% 
% function D = periodic_laplacian_1d(n)
%     D = sparse(n, n);
% 
%     % Main diagonal
%     D = D + sparse(1:n, 1:n, -2*ones(1,n), n, n);
% 
%     % Supra-diagonal
%     D = D + sparse(1:n-1, 2:n, ones(1,n-1), n, n);
% 
%     % Sub-diagonal  
%     D = D + sparse(2:n, 1:n-1, ones(1,n-1), n, n);
% 
%     % Periodic boundary terms
%     D(1, n) = 1;    % connects first to last
%     D(n, 1) = 1;    % connects last to first
% end
% 
% function D = dirichlet_laplacian_1d(n)
%     D = sparse(n, n);
% 
%     % Main diagonal
%     D = D + sparse(1:n, 1:n, -2*ones(1,n), n, n);
% 
%     % Supradiagonal
%     D = D + sparse(1:n-1, 2:n, ones(1,n-1), n, n);
% 
%     % Subdiagonal  
%     D = D + sparse(2:n, 1:n-1, ones(1,n-1), n, n);
% end

%% Laplacian U

main_diag = -2 * ones(Nx, 1);
off_diag = ones(Nx, 1);
D2_1d = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], Nx, Nx);

D2_1d_periodic = D2_1d;
D2_1d_periodic(1, end) = 1;      % top-right corner
D2_1d_periodic(end, 1) = 1;      % bottom-left corner

D2_1d_dirichlet = D2_1d;
D2_1d_dirichlet(1,1) = -3;
D2_1d_dirichlet(Nx,Nx) = -3;

D2_1d_periodic = D2_1d_periodic / dx2;
D2_1d_dirichlet = D2_1d_dirichlet / dx2;

I = speye(Ny);

Lap_U = kron(D2_1d_periodic, I) + kron(I, D2_1d_dirichlet);
spy(Lap_U);

%% Laplacian V 

main_diag = -2 * ones(Ny+1, 1);
off_diag = ones(Ny+1, 1);
D2_1d = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], Ny+1, Ny+1);

D2_1d_periodic = D2_1d;
D2_1d_periodic(1, end) = 1;      % top-right corner
D2_1d_periodic(end, 1) = 1;      % bottom-left corner

D2_1d = D2_1d / dx2;
D2_1d_periodic = D2_1d_periodic / dx2;

I = speye(Ny);

Lap_V = kron(D2_1d_periodic, I) + kron(I, D2_1d);
spy(Lap_V);

%% Gradient X

main_diag = ones(Nx, 1);
off_diag = -1 * ones(Nx, 1);
D2_1d = spdiags([off_diag, main_diag], [-1, 0], Nx, Nx);

D2_1d(1, end) = -1;      % top-right corner

D2_1d = D2_1d / dx;

I_y = speye(Ny);

G_x = kron(D2_1d, I_y);

spy(G_x);

%% Gradient Y

main_diag = ones(Ny+1, 1);
off_diag = -1 * ones(Ny+1, 1);
D2_1d = spdiags([main_diag, off_diag], [0, -1], Ny+1, Ny+1);

D2_1d = D2_1d / (dy);

I_x = speye(Nx);

G_y = kron(I_x, D2_1d);

G_y(:,1:Ny+1:end) = [];

% % Initialize sparse matrix for vertical derivative
% G_y = sparse(N_V, N_U); 
% 
% % Populate Dvx
% for i = 1:(N_V)
%     G_y(i, i) = -1; % Current pixel
%     G_y(i, i + Ny) = 1; % Pixel in the next row
% end
% 
% G_y = G_y / dy;

spy(G_y)

%% Divergence

D_x = -1 * G_x';
D_y = -1 * G_y';

% for k = 1:size(D_y, 1)
%     currentRow = D_y(k, :);
% 
%     % Check if all elements in the current row are the same
%     % This is true if the number of unique elements is 1
%     if nnz(currentRow) == 1
%         % If all elements are the same, set the entire row to zeros
%         D_y(k, :) = zeros(1, length(currentRow));
%     end
% end

spy(D_y);

%% Saddle Point System
Z1 = zeros(N_U, N_V);
Z2 = zeros(N_V, N_U);
A = [ Lap_U + sigma_U_diag     zeros(N_U, N_V)          G_x; 
      zeros(N_V, N_U)          Lap_V + sigma_V_diag     G_y;
      D_x                      D_y                      zeros(N_U, N_U) ];

RHS = [ f_bc(:);
        g_bc(:);
        h_bc(:); ];

sol = A\RHS;

U = sol(1:N_U);
V = sol(N_U+1:N_U+N_V);
P = sol(N_U+N_V+1:end);

Uplot = reshape(U, [Nx Ny]);
Vplot = reshape(V, [Nx + 1 Ny]);
Pplot = reshape(P, [Nx Ny]);

UFull = zeros(Nx+2,Ny+2);
UFull(2:Nx+1,2:Ny+1) = Uplot;
UFull(:,1) = UFull(:,Ny+1);
UFull(:,Ny+2) = NaN;
UFull = - UFull;

figure(1);
surf(X, Y, UFull);

% Vplot(2,:) = Vplot(2,:) + 3.5;
% Vplot(:,2) = Vplot(:,2) + 3.5;
% Vplot(:,Nx-1) = Vplot(:,Nx-1) + 3.5;
% Vplot = -Vplot;

% VFull = zeros(Nx+2,Ny+2);
% VFull(2:Nx+2,1:Ny) = Vplot;
% VFull(Ny+1,:) = VFull(2,:);
% VFull(1,:) = -3.5;
% VFull(Nx+2,:) = -3.5;

VFull = zeros(Nx+2,Ny+2);
VFull(1:Nx+1,2:Ny+1) = Vplot;
VFull(1,:) = 3.5;
VFull(Nx+2,:) = 3.5;
VFull(2,:) = VFull(2,:) + 3.5;
VFull(:,1) = VFull(:,Ny+1);
VFull(:,Ny+2) = VFull(:,2);
VFull = -VFull;
%VFull(:,1) = VFull(:,Ny+1);

figure(2);
surf(X, Y, VFull);

figure(3);
Umean = abs(sqrt(UFull.^2 + VFull.^2));
surf(X,Y,Umean);