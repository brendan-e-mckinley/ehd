%% Sophie Qual
clear; clc; close all;

%% Parameters
Nx = 64;           % interior points in x (pressure/U grid)
Ny = Nx;           % interior points in y
L = 1;
x = linspace(0, L, Nx+1);
dx = x(2) - x(1); 
dx2 = dx^2;
y = x; dy = dx;

xint = x(1:end-1);    % length Nx
yint = y(1:end-1);    % length Ny
y_j_U = yint + dx/2;
x_i_V = xint + dy/2; 

N_U = Nx * Ny;
Ny_minus = Ny - 1;
N_V = Nx * Ny_minus;
N_P = Nx * Ny;

%% BCs
V_bottom = -3.5;
V_top    = -3.5;
f_bc_mat = zeros(Nx, Ny);         
g_bc_mat = zeros(Nx, Ny_minus);   
h_bc_mat = zeros(Nx, Ny);    

f_bc = f_bc_mat(:);
g_bc_mat(:,1) = V_bottom / dy^2;
g_bc_mat(:,end) = V_top / dy^2;
g_bc = g_bc_mat(:);
h_bc_mat(:,1) = - V_bottom / dy;
h_bc_mat(:,end) =  V_top / dy;
h_bc = h_bc_mat(:);

%% Sigma
compute_sigma = @(xx, yy) 50 * (sin(2*pi*xx).*sin(4*pi*yy) + 1);

sigma_matrix_U = zeros(Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        sigma_matrix_U(i,j) = compute_sigma(xint(i), y_j_U(j));
    end
end
sigma_U_diag = spdiags(sigma_matrix_U(:), 0, N_U, N_U);

sigma_matrix_V = zeros(Nx, Ny_minus);
for i = 1:Nx
    for j = 1:Ny_minus
        sigma_matrix_V(i,j) = compute_sigma(x_i_V(i), yint(j) + dy);
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

%% Laplacians
Lap_U = kron(speye(Ny), D2_x) + kron(D2_y_U, speye(Nx));    % N_U x N_U
Lap_V = kron(speye(Ny_minus), D2_x) + kron(D2_y_V, speye(Nx)); % N_V x N_V

%% Gradients
D_x_forward = spdiags([-ones(Nx,1), ones(Nx,1)], [0 1], Nx, Nx);
D_x_forward(end,1) = 1;
D_x_forward = D_x_forward / dx;
G_x = kron(speye(Ny), D_x_forward);

D_y_small = spdiags([-ones(Ny_minus,1), ones(Ny_minus,1)], [0 1], Ny_minus, Ny);
D_y_small = D_y_small / dy;
G_y = kron(D_y_small, speye(Nx));

% Divergence
D_x = -G_x.';  % N_P x N_U
D_y = -G_y.';  % N_P x N_V

%% Saddle point system
A = [ Lap_U + sigma_U_diag     zeros(N_U, N_V)          G_x; 
      zeros(N_V, N_U)          Lap_V + sigma_V_diag     G_y;
      D_x                      D_y                      zeros(N_U, N_U) ];

RHS = [ f_bc;
        g_bc; 
        h_bc ];  

%% Solve
sol = A \ RHS;

% split
U = sol(1:N_U);
V = sol(N_U+1:N_U+N_V);
P = sol(N_U+N_V+1:end);

%% Plot
Uplot = reshape(U, [Nx, Ny]);
Vplot = reshape(V, [Nx, Ny_minus]);
Pplot = reshape(P, [Nx, Ny]);         

UFull = NaN(Nx+2, Ny+2);
UFull(2:Nx+1, 2:Ny+1) = Uplot;
UFull(:,1) = UFull(:,Nx);
UFull(:,Nx+1) = UFull(:,2);

VFull = V_bottom * ones(Nx+2, Ny+2);
VFull(2:Nx+1, 2:Ny) = -Vplot; 
VFull(:,1) = VFull(:,Nx); 
VFull(:,Nx+1) = VFull(:,2);

xplot = linspace(0,L,Nx+2);
yplot = linspace(0,L,Ny+2);
[Xplot, Yplot] = meshgrid(yplot, xplot);

figure; surf(Xplot, Yplot, UFull, 'EdgeColor','none'); title('U'); xlabel('y'); ylabel('x');
figure; surf(Xplot, Yplot, VFull, 'EdgeColor','none'); title('V'); xlabel('y'); ylabel('x');
figure; surf(Xplot, Yplot, sqrt(UFull.^2 + VFull.^2), 'EdgeColor','none'); title('|velocity|');

