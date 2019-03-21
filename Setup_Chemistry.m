function Setup_Chemistry()
disp("Setting up Chemistry")
global MW_Fuel MW_Ox n_moles_in_ox Aa Ea m n Hf Cp_mix n_Ox_n_F ig_start ig_stop dFdt rho U_in m_dot_Fuel1 m_dot_Fuel2 
global Pamb T_in M_dot_in h w phi1 frac_second dt

MW_Fuel = 44;   % C3H8
MW_Ox = 28.8;  % 28.8 for Air, 32 for O2
n_moles_in_ox = 4.76; % 4.76 for Air, 1 for O2

Aa = 8.6e11;
Ea = 15098;
m = 0.1;
n = 1.65;
Aa = Aa*1000^(1-m-n);
Hf = -2220; % kJ/mol
Cp_mix = @(T)min((1.9327e-10*T^4 - 7.9999e-7*T^3 +1.1407e-3*T^2 -4.489e-1*T + 1.0575e3),1300)/1000;

n_Ox_n_F = 3 + 8/4; % x + y/4 for CxHy
ig_start = 1e-4;
ig_stop = ig_start + 5*dt;

dFdt = @(AA,EEa,TT,FUEL,OXX)AA*exp(-EEa/TT)*(FUEL)^m*(OXX)^n;

rho = Pamb*MW_Ox/(8314*T_in);
U_in = M_dot_in/(rho*h*w);
m_dot_Fuel1 = phi1*M_dot_in*MW_Fuel/(MW_Ox*n_Ox_n_F*n_moles_in_ox);
m_dot_Fuel2 = m_dot_Fuel1*frac_second;

fprintf('Mass flow of Air: %2.2f g/s\n',M_dot_in*1000)
fprintf('Velocity of Air: %2.2f m/s\n',U_in)
fprintf('Mass flow of Fuel (at phi = %2.2f): %2.2f g/s\n',phi1,m_dot_Fuel1*1000)


end

