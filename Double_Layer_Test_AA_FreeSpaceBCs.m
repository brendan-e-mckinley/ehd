set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

clf
clear all

%%% Setup grid in x-y
Nx = 450; %256; % number of grid point along one direction
clf
L = 2.0*pi; %2.0*pi
x = linspace(-L/2,L/2,Nx+2); % periodic grid
dx = x(2)-x(1);
y = x;
dy = y(2)-y(1);

xint = x(2:end-1);
yint = y(2:end-1);
Ny = length(yint);

[X,Y] = meshgrid(x,y); % make 2D grid
[Xint,Yint] = meshgrid(xint,yint); % make 2D grid of interior points

%%% make finite difference laplacian

% dirichelet boundary conditions
e = (1/dy^2)*ones(Ny,1);
D2_d = spdiags([e -2*e e], -1:1, Ny, Ny);
I_nx = speye(Nx);
I_ny = speye(Ny);
Lap = kron(I_nx, D2_d) + kron(D2_d, I_ny);

%%%%%%%%%%%%%
beta_BC = 7.94;
sigma_bc = 0.78; %0.68
delta_layer = 0.1; %5*dx; %6*dx; %
%%%%%%%%%%%%%

%%%%%%%%%%% 
Phi_exact = @(x,y) beta_BC*y+0*x;
Npm_exact = @(x,y) 0*x+1.0;
Phi_BCs = 0*Xint;
Npm_BCs = 0*Xint;
Phi_BCs(1,:) = (1/dy/dy)*Phi_exact(xint,Y(1,1));
Phi_BCs(end,:) = Phi_BCs(end,:) + (1/dy/dy)*Phi_exact(xint,Y(end,end));
Phi_BCs(:,1) = Phi_BCs(:,1) + (1/dx/dx)*Phi_exact(X(1,1),yint');
Phi_BCs(:,end) = Phi_BCs(:,end) + (1/dx/dx)*Phi_exact(X(end,end),yint');
%%%%%%%%%%%%
Npm_BCs(1,:) = (1/dy/dy)*Npm_exact(xint,Y(1,1));
Npm_BCs(end,:) = Npm_BCs(end,:) + (1/dy/dy)*Npm_exact(xint,Y(end,end));
Npm_BCs(:,1) = Npm_BCs(:,1) + (1/dx/dx)*Npm_exact(X(1,1),yint');
Npm_BCs(:,end) = Npm_BCs(:,end) + (1/dx/dx)*Npm_exact(X(end,end),yint');
RHS = 0*Xint;

Test_Phi = Lap\(RHS(:)-Phi_BCs(:));
Test_Npm = Lap\(RHS(:)-Npm_BCs(:));
subplot(1,2,1)
surf(Xint,Yint,reshape(Test_Phi,Ny,Nx))
subplot(1,2,2)
surf(Xint,Yint,reshape(Test_Npm,Ny,Nx))
%%




colormap('turbo')

[sx,sy,sz] = sphere(50);

dLap = decomposition(Lap);
%%%%%%%%%%%%
% Make immersed boundary mats
rad = 0.25;
dth = dx/rad;
theta = (0:dth:(2*pi-dth))';
Nib = length(theta);
xib = rad*cos(theta);
yib = rad*sin(theta);
n_x = cos(theta);
n_y = sin(theta);
%%%%% Plot IB stuff
figure(1)
delta_a = @(r,a) ((1/(2*pi*a^2))*exp(-0.5*(r/a).^2)); 
delta = @(r) delta_a(r,1.2*dx);
delta_r = @(r) (1/(1.2*dx))^2*r.*delta_a(r,1.2*dx);

cut = 6*1.2*dx;
Sop_prime = @(q) spreadQ_prime(Xint,Yint,xib,yib,n_x,n_y,q,delta_r,cut);
Jop_prime = @(P) interpPhi_prime(Xint,Yint,xib,yib,n_x,n_y,P,delta_r,cut);
Sop = @(q) spreadQ(Xint,Yint,xib,yib,q,delta,cut);
Jop = @(P) interpPhi(Xint,Yint,xib,yib,P,delta,cut);
Phi_BC = Phi_exact(X,Y);
N_pm_BC = Npm_exact(X,Y);
G_d_G = @(Phi,N_pm) Grad_dot_Grad(Phi,N_pm,dx,dy,Nx,Ny,Phi_BC, N_pm_BC);
%%%%%%%%%%%%%
ctxt_BCs = [Phi_BCs(:);Npm_BCs(:);Npm_BCs(:);...
           0*xib(:)-(sigma_bc/delta_layer);0*xib(:);0*xib(:)];
AxOp_prev = @(ctxt, ctxt_prev) Constrained_Lap(ctxt, ctxt_prev, Lap, dLap, delta_layer, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime);
b_Op = @(ctxt) Build_RHS(ctxt, ctxt_BCs, Lap, dLap, G_d_G, delta_layer, dx, dy, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime);
%%%%%%%%%%%%% 
% Phi_init = Phi_exact(Xint,Yint);
% N_p_init = Npm_exact(Xint,Yint);
% N_m_init = Npm_exact(Xint,Yint);
% ctxt = [Phi_init(:);N_p_init(:);N_m_init(:);0*xib(:);0*xib(:);0*xib(:)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
ld = load('BC_run_N_300_r0p25.mat');
METHOD = 'makima'; %linear
EMETHOD = 'nearest'; %linear
Ny_ld = ld.Ny;
Nx_ld = ld.Nx;
Nib_ld = ld.Nib;
sz = Ny_ld*Nx_ld; 
Phi_ld = reshape(ld.ctxt(1:sz),Ny_ld,Nx_ld);
N_p_ld = reshape(ld.ctxt(sz+1:2*sz),Ny_ld,Nx_ld);
N_m_ld = reshape(ld.ctxt(2*sz+1:3*sz),Ny_ld,Nx_ld);
Q_ld = ld.ctxt(3*sz+1:3*sz+Nib_ld);
Q_p_ld = ld.ctxt(3*sz+Nib_ld+1:3*sz+2*Nib_ld);
Q_m_ld = ld.ctxt(3*sz+2*Nib_ld+1:3*sz+3*Nib_ld);
%%%
Phi_init_f = griddedInterpolant(ld.Xint',ld.Yint',Phi_ld',METHOD,'linear');
N_p_init_f = griddedInterpolant(ld.Xint',ld.Yint',N_p_ld',METHOD,EMETHOD);
N_m_init_f = griddedInterpolant(ld.Xint',ld.Yint',N_m_ld',METHOD,EMETHOD);

Phi_init = Phi_init_f(Xint',Yint')';
N_p_init = N_p_init_f(Xint',Yint')';
N_m_init = N_m_init_f(Xint',Yint')';
% theta_ld = atan2(ld.yib,ld.xib);
% theta_ld = theta_ld .* (theta_ld >= 0) + (2 * pi + theta_ld) .* (theta_ld < 0);
theta_ld = ld.theta;
Q_init = interp1(theta_ld,Q_ld,theta,METHOD,'extrap');
Q_p_init = interp1(theta_ld,Q_p_ld,theta,METHOD,'extrap');
Q_m_init = interp1(theta_ld,Q_m_ld,theta,METHOD,'extrap');

ctxt = [Phi_init(:);N_p_init(:);N_m_init(:);Q_init(:);Q_p_init(:);Q_m_init(:)];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% ctxt_gmres = ctxt;

RHS = b_Op(ctxt);
err = norm(AxOp_prev(ctxt,ctxt)-RHS)/norm(RHS);
disp(['residual: ' num2str(err)])

beta = 0.2;
m = 50;
DU = NaN(length(RHS),m);
DG = NaN(length(RHS),m);


tol = 1e-5; %0.001*dx*dx; %0.01*dx*dx
u_n = ctxt;
RHS = b_Op(ctxt);
AxOp = @(xx) AxOp_prev(xx,ctxt);
[G_u_n,FLAG,RELRES,ITER] = gmres(AxOp,RHS(:),1000,tol,1000,[],[],u_n); 
u_next = G_u_n;
G_u_next = G_u_n;
for its = 1:100000
RHS = b_Op(u_next);
AxOp = @(xx) AxOp_prev(xx,u_next);
[G_u_next,FLAG,RELRES,ITER,Resvec] = gmres(AxOp,RHS(:),1000,tol,1000,[],[],u_next); 

m_n = min(m,its);
DU(:,m_n) = u_next-u_n;
DG(:,m_n) = G_u_next - G_u_n;
if size(DU,2) > m
    DU = DU(:,2:end);
    DG = DG(:,2:end);
end

f_n = G_u_next - u_next;
DF = DG(:,1:m_n) - DU(:,1:m_n);
[Qdf,Rdf] = qr(DF,"econ");
gamma = Rdf\(Qdf'*f_n);
u_n = u_next;
G_u_n = G_u_next;

u_next = (G_u_next - DG(:,1:m_n)*gamma) - (1-beta)*(f_n-DF*gamma);


Phi = reshape(u_next(1:Ny*Nx),Ny,Nx);
Np = reshape(u_next(Ny*Nx+1:2*Nx*Ny),Ny,Nx);
Nm = reshape(u_next(2*Ny*Nx+1:3*Nx*Ny),Ny,Nx);

RHS = b_Op(u_next);
err(its) = norm(AxOp_prev(u_next,u_next)-RHS)/norm(RHS);
disp(['residual: ' num2str(err(its))])
disp(['Itts: ' num2str(ITER(end))])
if err(its) < 1e-4 %1e-3
    disp('done')
    break
end

clf
hs = surf(Xint,Yint,Np);
set(hs,'facealpha',0.8)
hold all
xlabel('x')
ylabel('y')
title('potential  $$\phi$$')
drawnow
end

ctxt = u_next;
%save('Test_run_N_256.mat','ctxt','Xint','Yint','xib','yib')

save('Err_Run_N_450.mat','ctxt','u_next','Xint','Yint','xib','yib','Phi','Np','Nm','err','Resvec','ITER')



%%
function A_x_Ctx = Constrained_Lap(ctxt, ctxt_prev, Lap, dLap, delta_layer, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime)
    A_x_Ctx = 0*ctxt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sz = Nx*Ny;
    Phi = ctxt(1:sz);
    N_p = ctxt(sz+1:2*sz);
    N_m = ctxt(2*sz+1:3*sz);
    q_i = 3*sz;
    Q = ctxt((q_i+1):(q_i+Nib));
    Q_p = ctxt((q_i+Nib+1):(q_i+2*Nib));
    Q_m = ctxt((q_i+2*Nib+1):(q_i+3*Nib));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Phi_prev = ctxt_prev(1:sz);
    N_p_prev = ctxt_prev(sz+1:2*sz);
    N_m_prev = ctxt_prev(2*sz+1:3*sz);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SQ = Sop_prime(Q);
    SQ_p = Sop_prime(Q_p);
    SQ_m = Sop_prime(Q_m);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dl2 = delta_layer^2;
    A_x_Ctx(1:sz) = dl2*Phi + dLap\(0.5*N_p - 0.5*N_m + SQ(:));
    A_x_Ctx(sz+1:2*sz) =  N_p + dLap\SQ_p(:); %N_p_prev.*(Lap*Phi) +
    A_x_Ctx(2*sz+1:3*sz) = N_m + dLap\SQ_m(:); %-N_m_prev.*(Lap*Phi) + 
    A_x_Ctx((q_i+1):(q_i+Nib)) = Jop_prime(reshape(Phi,Ny,Nx));
    A_x_Ctx((q_i+Nib+1):(q_i+2*Nib)) = Jop_prime(reshape(N_p,Ny,Nx));
    A_x_Ctx((q_i+2*Nib+1):(q_i+3*Nib)) = Jop_prime(reshape(N_m,Ny,Nx));
end

function b_Ctx = Build_RHS(ctxt, ctxt_BCs, Lap, dLap, Grad_dot_Grad, delta_layer, dx, dy, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime)
    % ctxt_BCs: has all of the Stuff in the RHS that 
    % does *not* change from iteration to iteration
    b_Ctx = 0*ctxt_BCs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sz = Nx*Ny;
    q_i = 3*sz;
    Phi = ctxt(1:sz);
    N_p = ctxt(sz+1:2*sz);
    N_m = ctxt(2*sz+1:3*sz);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Phi_BC = ctxt_BCs(1:sz);
    N_p_BC = ctxt_BCs(sz+1:2*sz);
    N_m_BC = ctxt_BCs(2*sz+1:3*sz);
    Q_BC = ctxt_BCs((q_i+1):(q_i+Nib));
    Q_p_BC = ctxt_BCs((q_i+Nib+1):(q_i+2*Nib));
    Q_m_BC = ctxt_BCs((q_i+2*Nib+1):(q_i+3*Nib));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dl2 = delta_layer^2;
    b_Ctx(1:sz) = dLap\(-dl2*Phi_BC(:));
    % N_BCs are = 1 so no need to include them in the following lines
    b_Ctx(sz+1:2*sz) = dLap\(-N_p(:).*(Lap*Phi(:)+Phi_BC(:)) - N_p_BC(:) - Grad_dot_Grad(Phi,N_p));
    b_Ctx(2*sz+1:3*sz) = dLap\(N_m(:).*(Lap*Phi(:)+Phi_BC(:)) - N_m_BC(:) + Grad_dot_Grad(Phi,N_m));
    b_Ctx((q_i+1):(q_i+Nib)) = Q_BC;
    b_Ctx((q_i+Nib+1):(q_i+2*Nib)) = Q_p_BC - Jop(reshape(N_p,Ny,Nx)).*Jop_prime(reshape(Phi,Ny,Nx));
    b_Ctx((q_i+2*Nib+1):(q_i+3*Nib)) = Q_m_BC + Jop(reshape(N_m,Ny,Nx)).*Jop_prime(reshape(Phi,Ny,Nx));
end

function G_d_G = Grad_dot_Grad(Phi,N_pm,dx,dy,Nx,Ny,Phi_BC, N_pm_BC)
    % Phi_x * N_x + Phi_y * N_y
    Phi = reshape(Phi,Ny,Nx);
    N_pm = reshape(N_pm,Ny,Nx);
    Phi_BC_y = [Phi_BC(1,2:end-1); Phi; Phi_BC(end,2:end-1)];
    Phi_y = (0.5/dy)*(Phi_BC_y(3:end,:)-Phi_BC_y(1:end-2,:));
    N_pm_BC_y = [N_pm_BC(1,2:end-1); N_pm; N_pm_BC(end,2:end-1)];
    N_pm_y = (0.5/dy)*(N_pm_BC_y(3:end,:)-N_pm_BC_y(1:end-2,:));
    Phi_BC_x = [Phi_BC(2:end-1,1), Phi, Phi_BC(2:end-1,end)];
    Phi_x = (0.5/dx)*(Phi_BC_x(:,3:end)-Phi_BC_x(:,1:end-2));
    N_pm_BC_x = [N_pm_BC(2:end-1,1), N_pm, N_pm_BC(2:end-1,end)];
    N_pm_x = (0.5/dx)*(N_pm_BC_x(:,3:end)-N_pm_BC_x(:,1:end-2));
    G_d_G = N_pm_x.*Phi_x + N_pm_y.*Phi_y;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G_d_G = G_d_G(:);
end

% Dipole spreading operation
function [Sq] = spreadQ_prime(X,Y,xq,yq,n_x,n_y,q,delta_r,cut)
    Sq = 0*X;
    Nq = length(q);
    for k = 1:Nq
        Rk = sqrt((X-xq(k)).^2 + (Y-yq(k)).^2);
        mask = (Rk <= cut);
        n_dot_rhat = (n_x(k)*(X(mask)-xq(k)) + n_y(k)*(Y(mask)-yq(k)))./Rk(mask);
        Sq(mask) = Sq(mask) + q(k)*n_dot_rhat.*delta_r(Rk(mask));
    end
end

% Dipole interpolation operation
function [Jphi] = interpPhi_prime(X,Y,xq,yq,n_x,n_y,Phi,delta_r,cut)
    Jphi = 0*xq;
    Nq = length(xq);
    dx = X(1,2)-X(1,1);
    dy = Y(2,1)-Y(1,1);
    for k = 1:Nq
        Rk = sqrt((X-xq(k)).^2 + (Y-yq(k)).^2);
        mask = (Rk <= cut);
        n_dot_rhat = (n_x(k)*(X(mask)-xq(k)) + n_y(k)*(Y(mask)-yq(k)))./Rk(mask);
        Jphi(k) = dx*dy*sum(sum(Phi(mask).*n_dot_rhat.*delta_r(Rk(mask))));
    end
end

% Charge spreading operation
function [Sq] = spreadQ(X,Y,xq,yq,q,delta,cut)
    Sq = 0*X;
    Nq = length(q);
    for k = 1:Nq
        Rk = sqrt((X-xq(k)).^2 + (Y-yq(k)).^2);
        mask = (Rk <= cut);
        Sq(mask) = Sq(mask) + q(k)*delta(Rk(mask));
    end
end

% Charge interpolation operation
function [Jphi] = interpPhi(X,Y,xq,yq,Phi,delta,cut)
    Jphi = 0*xq;
    Nq = length(xq);
    dx = X(1,2)-X(1,1);
    dy = Y(2,1)-Y(1,1);
    for k = 1:Nq
        Rk = sqrt((X-xq(k)).^2 + (Y-yq(k)).^2);
        mask = (Rk <= cut);
        Jphi(k) = dx*dy*sum(sum(Phi(mask).*delta(Rk(mask))));
    end
end