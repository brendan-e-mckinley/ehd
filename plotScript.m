set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

clear;clc;

ld = load('Err_Run_Schur.mat');

Phi = ld.Phi';
Np = ld.Np';
Nm = ld.Nm';
Xint = ld.Xint;
Yint = ld.Yint;

% grid
Nx = 450;
L = 2.0*pi;
x = linspace(-L/2,L/2,Nx+2);
dx = x(2)-x(1);

% immersed boundary 
rad = 0.25 + .001;
dth = dx/rad;
theta = (0:dth:(2*pi-dth))';
Nib = length(theta);
xib = rad*cos(theta);
yib = rad*sin(theta);
rad_cover = rad - .002;
xib_cover = rad_cover*cos(theta);
yib_cover = rad_cover*sin(theta);

% delta layer
delta = .1;
rad_delta = 0.25 + delta;
xib_delta = rad_delta*cos(theta);
yib_delta = rad_delta*sin(theta);
xib_closed = [xib_delta; xib_delta(1)];
yib_closed = [yib_delta; yib_delta(1)];

% color
customColor_Np = [237 232 208] / 255;
customColor_Nm = [237 232 208] / 255;

fig1 = figure(1);
set(fig1, 'Position', [100 100 600 600]);  % square figure window

Np_plot = surf(Xint,Yint,Np);
set(Np_plot,'facealpha',0.8);
xlabel('x'); ylabel('y');
xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
title('$$n_+$$','Interpreter','latex');
colormap(slanCM('spring'));
colorbar;
view(0, 90)
shading interp
hold on 
fill3(xib_cover, yib_cover, zeros(size(xib_cover)) + 2, customColor_Np, 'FaceAlpha', 1, 'EdgeColor', 'none');
plot3(xib, yib, zeros(size(xib)) + 2, 'k.', 'MarkerSize', 6);
plot3(xib_closed, yib_closed, zeros(size(xib_closed)) + 2, '-', 'Color', customColor_Np, 'LineWidth', 2);

axis equal          % equal scaling x:y
pbaspect([1 1 1])   % ensures the plot box is a cube (even if z is different)
hold off

fig2 = figure(2);
set(fig2, 'Position', [100 100 600 600]);  % same square size

Nm_plot = surf(Xint,Yint,Nm);
set(Nm_plot,'facealpha',0.8);
xlabel('x'); ylabel('y');
xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
title('$$n_-$$','Interpreter','latex');
colormap(slanCM('bone'));
colorbar; caxis([0.9 1.3]);
view(0, 90)
shading interp
hold on 
fill3(xib_cover, yib_cover, zeros(size(xib_cover)) + 2, customColor_Nm, 'FaceAlpha', 1, 'EdgeColor', 'none');
plot3(xib, yib, zeros(size(xib)) + 2, 'k.', 'MarkerSize', 6);
plot3(xib_closed, yib_closed, zeros(size(xib_closed)) + 2, '-', 'Color', customColor_Np, 'LineWidth', 2);

axis equal
pbaspect([1 1 1])
hold off

figure(3);
Phi_plot = surf(Xint,Yint,Phi);
set(Phi_plot,'facealpha',0.8);
xlabel('x');
ylabel('y');
xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
title('$$\phi$$');
colormap(slanCM('thermal-2'));
colorbar; caxis([-3 3]);
view(0, 90)
shading interp
pbaspect([1 1 1])