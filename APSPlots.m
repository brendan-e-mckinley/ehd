%% Immersed Boundary Method
clear; clc; close all;
% 
% set(0,'defaulttextInterpreter','latex')
% set(0,'defaultAxesTickLabelInterpreter','latex'); 
% set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',0.05);
% set(0,'defaultAxesFontSize',35)

Nx = 450; %256; % number of grid point along one direction
L = 2.0*pi; %2.0*pi
x = linspace(-L/2,L/2,Nx+2); % periodic grid
dx = x(2)-x(1);
y = x;
dy = y(2)-y(1);

[X,Y] = meshgrid(x,y); % make 2D grid
X_scaled = X / 0.25;
Y_scaled = Y / 0.25;

%%%%%%%%%%%%
% Make immersed boundary mats
rad = 0.25;
dth = dx/rad;
theta = (0:dth:(2*pi))';
Nib = length(theta);
xib = rad*cos(theta);
yib = rad*sin(theta);
n_x = cos(theta);
n_y = sin(theta);
xib_scaled = xib / 0.25;
yib_scaled = yib / 0.25;

X_i = 0.919491;
Y_i = 0.975217;
x_i = 0.67051;
y_i = 0.741901;

figure;
plot(X_scaled, Y_scaled, 'r-', X_scaled', Y_scaled', 'r-'); % Plots horizontal and vertical lines in blue
set(gca, 'Color', [0.8 0.9 1.0]); 
hold on;
patch(xib_scaled, yib_scaled, [1 1 1]); 

xib_scaled(16) = NaN;
find_xib = xib_scaled - x_i;
find_Xib = X_scaled - X_i;
zero_locations = (find_Xib == 0);
X_scaled(zero_locations) = NaN;
Y_scaled(zero_locations) = NaN;

scatter(xib_scaled,yib_scaled,'filled', 'MarkerFaceColor', 'b')
scatter(x_i,y_i,'filled', 'MarkerFaceColor', 'm')
scatter(X_i,Y_i,'s','filled', 'MarkerFaceColor', 'm')
xlabel('x / R');
ylabel('y / R');
xlim([-1.25, 1.25]);
ylim([-1.25, 1.25]);
title('2D Meshgrid as a Grid');
hold off;

%% Regularized Deltas
set(0,'defaultLineLineWidth',5);
set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesFontSize',15)

delta_a = @(r,a) ((1/(2*pi*a^2))*exp(-0.5*(r/a).^2)); 
delta = @(r) delta_a(r,1.2*dx);
delta_r = @(r) (1/(1.2*dx))^2*r.*delta_a(r,1.2*dx);

plot_r = linspace(-0.05,0.05,Nx+2);

figure;

% Create the first (top) plot in a 2x1 grid, position 1
ax1 = subplot(2, 1, 1);
plot(ax1, plot_r, delta(plot_r));
xlabel(ax1, '$\mathbf{x}_i - \mathbf{X}_g$', 'Interpreter', 'latex');
ylabel(ax1, '$\delta_h$', 'Interpreter', 'latex');
ax1.YLabel.Rotation = 90;
%title(ax1, '$\delta_h(\mathbf{x}_i - \mathbf{X}_g)$', 'Interpreter', 'latex');

% Create the second (bottom) plot in a 2x1 grid, position 2
ax2 = subplot(2, 1, 2);
plot(plot_r, delta_r(plot_r));
xlabel(ax2, '$\mathbf{x}_i - \mathbf{X}_g$', 'Interpreter', 'latex');
ylabel(ax2, '$\frac{\partial\delta_h}{\partial\nu_i}$', 'Interpreter', 'latex');
ax2.YLabel.Rotation = 90;
ylim(ax2, [-25000, 25000])
%title(ax2, '$\frac{\partial\delta_h}{\partial\nu_i}(\mathbf{x}_i - \mathbf{X}_g)$', 'Interpreter', 'latex');


