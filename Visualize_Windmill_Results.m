set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

clear;clc;

ld = load('Windmill_Results_Plot_Grant.mat');
ld_geom = load('Windmill_Geom_Larger_Next.mat');

ctxt_ld = ld.ctxt;

Phi = ld.Phi;
Np = ld.Np;
Nm = ld.Nm;
Xint = ld.Xint;
Yint = ld.Yint;

UFull = ld.UFull;
VFull = ld.VFull;
X = ld.X;
Y = ld.Y;

% grid
Nx = ld.Nx;
L = 4.0 * pi;
x = linspace(-L/2,L/2,Nx+2);
dx = x(2)-x(1);

% immersed boundary 
xib = ld.xib;
yib = ld.yib;
Nib = ld.Nib;

% delta layer
x_finest = linspace(-L/2, L/2, 450);
dx_finest = x_finest(2) - x_finest(1);
beta_BC = 7.94;
delta_layer = 0.1 * ((5 * dx) / (5*dx_finest));

% color
customColor_Np = [237 232 208] / 255;
customColor_Nm = [237 232 208] / 255;

% fig1 = figure(1);
% set(fig1, 'Position', [100 100 600 600]);  % square figure window
% 
% Np_plot = surf(Xint,Yint,Np);
% set(Np_plot,'facealpha',0.8);
% xlabel('x'); ylabel('y');
% xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
% title('$$n_+$$','Interpreter','latex');
% colormap(slanCM('spring'));
% colorbar;
% view(0, 90)
% shading interp
% hold on 
% fill3(xib_cover, yib_cover, zeros(size(xib_cover)) + 2, customColor_Np, 'FaceAlpha', 1, 'EdgeColor', 'none');
% plot3(xib, yib, zeros(size(xib)) + 2, 'k.', 'MarkerSize', 6);
% plot3(xib_closed, yib_closed, zeros(size(xib_closed)) + 2, '-', 'Color', customColor_Np, 'LineWidth', 2);
% 
% axis equal          % equal scaling x:y
% pbaspect([1 1 1])   % ensures the plot box is a cube (even if z is different)
% hold off
% 
% fig2 = figure(2);
% set(fig2, 'Position', [100 100 600 600]);  % same square size
% 
% Nm_plot = surf(Xint,Yint,Nm);
% set(Nm_plot,'facealpha',0.8);
% xlabel('x'); ylabel('y');
% xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
% title('$$n_-$$','Interpreter','latex');
% colormap(slanCM('bone'));
% colorbar; caxis([0.9 1.3]);
% view(0, 90)
% shading interp
% hold on 
% fill3(xib_cover, yib_cover, zeros(size(xib_cover)) + 2, customColor_Nm, 'FaceAlpha', 1, 'EdgeColor', 'none');
% plot3(xib, yib, zeros(size(xib)) + 2, 'k.', 'MarkerSize', 6);
% plot3(xib_closed, yib_closed, zeros(size(xib_closed)) + 2, '-', 'Color', customColor_Np, 'LineWidth', 2);
% 
% Npm = Np - Nm;
% 
% fig3 = figure(3);
% set(fig3, 'Position', [100 100 600 600]);  % same square size
% Npm_plot = surf(Xint,Yint,Npm);
% set(Npm_plot,'facealpha',0.8);
% xlabel('x'); ylabel('y');
% xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
% title('$$n_+ - n_-$$','Interpreter','latex');
% colormap(slanCM('spring'));
% colorbar; caxis([0.9 1.3]);
% view(0, 90)
% shading interp
% hold on 
% fill3(xib_cover, yib_cover, zeros(size(xib_cover)) + 2, customColor_Nm, 'FaceAlpha', 1, 'EdgeColor', 'none');
% plot3(xib, yib, zeros(size(xib)) + 2, 'k.', 'MarkerSize', 6);
% plot3(xib_closed, yib_closed, zeros(size(xib_closed)) + 2, '-', 'Color', customColor_Np, 'LineWidth', 2);
% 
% axis equal
% pbaspect([1 1 1])
% hold off

disturbance = Phi - beta_BC * Yint;
%[Field_X, Field_Y] = gradient(disturbance);
[Field_X, Field_Y] = gradient(Phi);
mask = (Xint >= min(xib) - 0.05 & Xint <= max(xib) + 0.05) & (Yint >= min(yib) - 0.05 & Yint <= max(yib) + 0.05);
% Field_X(mask) = NaN;
% Field_Y(mask) = NaN;
R  = 0.25;
rescaled_X = X / R;
rescaled_Y = Y / R;

figure(1);
Phi_plot = surf(Xint / R,Yint / R, disturbance);
% lines = streamslice(Xint / R,Yint / R, Field_X, Field_Y,'noarrows');
% % Set the lines to dashed
% set(lines,'LineStyle','--','Color','k');
% 
% % Lift the streamline to the top (ensure it is above max Z)
% for i=1:length(lines)
%     lines(i).ZData = ones(size(lines(i).XData)) * max(Phi(:)) + 1;
% end

set(Phi_plot,'facealpha',0.8);
xlabel('x/a');
ylabel('y/a');
%xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
title('$$\Phi - \beta y$$');
%colormap(slanCM('thermal-2'));
%colorbar; caxis([-3 3]);
view(0, 90)
shading interp
pbaspect([1 1 1])
hold on;

% Parameters
h = sqrt(3)*R;
shift = 0.05;
sep = 1.4;
centers = [
    sep * (-R - shift),   sep * (-h/2);        % bottom-left
    sep * ( R + shift),   sep * (-h/2);        % bottom-right
    sep * ( 0.0 ),        sep * ( h/2 + shift) % top-middle
];

% For z-offset above Phi surface (set before calling this snippet)
if exist('Phi','var')
    ztop = max(Phi(:)) + 1; 
else
    ztop = 1;  % fallback if Phi not defined yet
end

% Helper for drawing filled circles as patches
drawCircle = @(cx,cy,R) ...
    patch(cx + R*cos(linspace(0,2*pi,200)), ...
          cy + R*sin(linspace(0,2*pi,200)), ...
          ztop*ones(1,200), ...
          'k', 'EdgeColor','none');

% Circle 1 (bottom-left)
p1 = drawCircle(centers(1,1) / R, centers(1,2) / R, 1);

% Circle 2 (bottom-right)
p2 = drawCircle(centers(2,1) / R, centers(2,2) / R, 1);

% Circle 3 (top-middle)
p3 = drawCircle(centers(3,1) / R, centers(3,2) / R, 1);

% Circle 4
p4 = drawCircle(0.2 / R, 0.05 / R, 0.1 / R);

% Circle 5
p5 = drawCircle(-0.2 / R, 0.05 / R, 0.1 / R);

% Circle 6
p6 = drawCircle(0, -0.2 / R, 0.1 / R);

% Collect handles (optional)
r = [p1 p2 p3 p4 p5 p6];

hold off;

figure(2);
a= R;
N_net_plot = surf(Xint/a,Yint/a, Np - Nm);
lines = streamslice(Xint / R,Yint / R, Field_X, Field_Y,2,'noarrows');
% Set the lines to dashed
set(lines,'LineStyle','--','LineWidth',1.0,'Color','k');

% Lift the streamline to the top (ensure it is above max Z)
for i=1:length(lines)
    lines(i).ZData = ones(size(lines(i).XData)) * max(Phi(:)) + 1;
end
set(N_net_plot,'facealpha',0.8);
xlabel('x/a');
ylabel('y/a');
xlim([-10, 10]);
ylim([-10, 10]);
%xlim([-pi/6 pi/6]); ylim([-pi/6 pi/6]);
title('$$N_p - N_m$$');
colormap(slanCM('bwr'));
colorbar; caxis([-3 3]);
view(0, 90)
shading interp
pbaspect([1 1 1])
hold on;

%UFull(mask) = NaN;
%VFull(mask) = NaN;
vel_lines = streamslice(X / a, Y / a, UFull, VFull);
set(vel_lines,'LineWidth',1.0,'Color','k');
% Lift the streamline to the top (ensure it is above max Z)
for i=1:length(vel_lines)
    vel_lines(i).ZData = ones(size(vel_lines(i).XData)) * max(Phi(:)) + 1;
end

% Subsample for fewer arrows
% skip = 5;
% Xs = rescaled_X(1:skip:end, 1:skip:end);
% Ys = rescaled_Y(1:skip:end, 1:skip:end);
% Us = UFull(1:skip:end, 1:skip:end);
% Vs = VFull(1:skip:end, 1:skip:end);
% % Overlay quiver in 2D
% quiver(Xs, Ys, Us, Vs, 1.5, 'k', 'LineWidth', 1.0);  % black, thicker, shorter
% xlabel('x/a'); ylabel('y/a');
% title('|velocity|');

%scatter(ld_geom.x / a, ld_geom.y / a, 40, 'k', 'filled')

% Circle 1 (bottom-left)
p1 = drawCircle(centers(1,1)/a, centers(1,2)/a, 1);

% Circle 2 (bottom-right)
p2 = drawCircle(centers(2,1)/a, centers(2,2)/a, 1);

% Circle 3 (top-middle)
p3 = drawCircle(centers(3,1)/a, centers(3,2)/a, 1);

% Circle 4
p4 = drawCircle(0.2/a, 0.05/a, 0.1/a);

% Circle 5
p5 = drawCircle(-0.2/a, 0.05/a, 0.1/a);

% Circle 6
p6 = drawCircle(0, -0.2/a, 0.1/a);
hold off;

% Compute velocity magnitude
Vmag = sqrt(UFull.^2 + VFull.^2);

figure(3);
% Plot surface for magnitude (semi-transparent)
hSurf = surf(rescaled_X, rescaled_Y, Vmag);
set(hSurf,'FaceAlpha',0.6);   % makes overlay visible
shading interp;
colormap(slanCM('turbo'));
xlabel('x/a');
ylabel('y/a');
colorbar;
hold on;
axis equal;
view(0,90);

% Subsample for fewer arrows
skip = 5;
Xs = rescaled_X(1:skip:end, 1:skip:end);
Ys = rescaled_Y(1:skip:end, 1:skip:end);
Us = UFull(1:skip:end, 1:skip:end);
Vs = VFull(1:skip:end, 1:skip:end);
% Overlay quiver in 2D
quiver(Xs, Ys, Us, Vs, 1.5, 'k', 'LineWidth', 1.0);  % black, thicker, shorter
xlabel('x/a'); ylabel('y/a');
title('|velocity|');

% Helper for 2D filled circles
drawCircle2D = @(cx,cy,R) patch(cx + R*cos(linspace(0,2*pi,200)), ...
                                cy + R*sin(linspace(0,2*pi,200)), ...
                                'k', 'EdgeColor','none');

% Draw circles
drawCircle2D(centers(1,1) / a, centers(1,2) / a, 1);
drawCircle2D(centers(2,1) / a, centers(2,2) / a, 1);
drawCircle2D(centers(3,1) / a, centers(3,2) / a, 1);
drawCircle2D(0.2 / a, 0.05 / a, 0.1 / a);
drawCircle2D(-0.2 / a, 0.05 / a, 0.1 / a);
drawCircle2D(0, -0.2 / a, 0.1 / a);

