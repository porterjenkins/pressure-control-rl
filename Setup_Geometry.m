function [ output_args ] = Setup_Geometry(pos_primary,pos_secondary,pos_ignition)

global L h w xi1 xi2 xp ig_pos dx divs x dt

L = 1.5;    % Length of Combustor (in m)
h = 0.1;   % Height of Combustor (in m)
w = 0.1;   % Width of Combustor (in m)
xi1 = pos_primary;   % Location of Primary Injector (in m)
xi2 = pos_secondary;    % Location of Control Injector (in m)
xp = 1.0;   % Sensor Location (in m)
ig_pos = pos_ignition; % Location of Primary Ignitor (in m)
%dx = 0.006;
%divs = ceil(L/dx); % Spatial Divisions

divs = 100;
dx = L/divs;
x = (1:divs)*dx;

fprintf('Max Velocity : %2.2f \n\n',dx/dt);

end

