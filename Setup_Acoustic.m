function Setup_Acoustic(damp_coeff)
disp("Setting up Acoustics")
global N c gamma zeta1 nn kn phi dphi
global L

N = 10;         % Number of modes
c = 330;        % Speed of Sound
gamma = 1.4;    % Gamma of Air
zeta1 = damp_coeff;  % Damping Coefficient
nn = 1:N;

% %Closed-Open
% 
% kn = 0.5*pi*(1:2:2*N-1)/L;  % Wave numbers for N modes
% phi = @(xx)sin(kn*xx);
% dphi = @(xx)kn.*cos(kn*xx);

%Open-Closed

kn = 0.5*pi*(1:2:2*N-1)/L;  % Wave numbers for N modes
phi = @(xx)cos(kn*xx);
dphi = @(xx)-kn.*sin(kn*xx);

end

