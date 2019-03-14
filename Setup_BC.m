function Setup_BC(total_time,Mass_in,phi_primary,frac_sec)

global ttot dt M_dot_in T_in Pamb ig_temp phi1 frac_second

ttot = total_time;
dt = 1e-4;      % Time Step
M_dot_in = Mass_in;
M_dot_in = M_dot_in/1000;
T_in = 300;     % Inlet Temp
Pamb = 101325;  % Ambient Pressure
ig_temp = 1300;
phi1 = phi_primary;   % Phi of primary burner
frac_second = frac_sec;

end

