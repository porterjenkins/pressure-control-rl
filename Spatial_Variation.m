%close all;
clear all;
clc;

%% Geometry Setup

L = 1;    % Length of Combustor (in m)
xi1 = 0.2;   % Location of Primary Injector (in m)
xi2 = 1.3;    % Location of Control Injector (in m)
xp = 1.0;   % Sensor Location (in m)

dx = 0.003;
divs = ceil(L/dx); % Spatial Divisions
x = (0:divs)*dx;
%divs = 450; % Spatial Divisions

%% Acoustic Setup

N = 20;         % Number of modes

%% Set Acoustic BCs

% Closed == Constant Pressure
% Open == Constant Velocity

inlet = 0;      % 1 = closed, 0 = open
outlet = 1;     % 1 = closed, 0 = open

if inlet == 0 & outlet == 0
    %Open-Open
    kn = (1:N)*pi/L;
    phi = @(xx)cos(kn*xx);
    dphi = @(xx)-kn.*sin(kn*xx);
elseif inlet == 1 & outlet == 1
    %Closed-Closed
    kn = (1:N)*pi/L;
    phi = @(xx)sin(kn*xx);
    dphi = @(xx)kn.*cos(kn*xx);
elseif inlet == 1 & outlet == 0
    %Closed-Open
    kn = 0.5*pi*(1:2:2*N-1)/L;  % Wave numbers for N modes
    phi = @(xx)sin(kn*xx);
    dphi = @(xx)kn.*cos(kn*xx);
elseif inlet == 0 & outlet == 1
    %Open-Closed
    kn = 0.5*pi*(1:2:2*N-1)/L;  % Wave numbers for N modes
    phi = @(xx)cos(kn*xx);
    dphi = @(xx)-kn.*sin(kn*xx);
end

for i=1:length(x)
    sv_p(i) = sum(phi(x(i)));
    sv_u(i) = sum(dphi(x(i)));
end

% plot(x,sv_p,x,sv_u);
% legend('Pressure','Velocity');

plot(x,sv_p);
xlim([0 L]);
legend('Pressure');





