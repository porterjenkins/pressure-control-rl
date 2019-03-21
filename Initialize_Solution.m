function Initialize_Solution()

global totalsteps X A cells Q_dot f1_index f2_index
global N divs xi1 xi2 M_dot_in L U_in n_moles_in_ox MW_Ox m_dot_Fuel1 T_in Cp_mix ttot dt

%totalsteps = ceil(ttot/dt);         % Total time steps
%totalsteps = 1000

X = zeros(2*N,1);
X(1,1) = 0/101325;       % Initial amplitude for 1st mode

A = zeros(2*N,2*N);

cells = zeros(5,divs);  % Index 1 contains mass of oxidizer
                        % Index 2 contains mass of fuel
                        % Index 3 contains mass of total gas mass
                        % Index 4 contains heat content
                        % Index 5 contains temperature

Q_dot = zeros(1,divs);

f1_index = ceil(xi1/L*divs);
f2_index = ceil(xi2/L*divs);

cells(1,:) = M_dot_in*L/(U_in*divs)*32/(n_moles_in_ox*MW_Ox);  % kg of oxidizer
cells(2,[f1_index:size(cells,2)]) = m_dot_Fuel1*(L-xi1)/(U_in*(divs-f1_index));  % kg of fuel
cells(3,:) = M_dot_in*L/(U_in*divs) + cells(2,:);   % kg total (Air + Fuel + Burnt Gases)
cells(5,:) = T_in;  % Kelvin
cells(4,:) = cells(3,:)*Cp_mix(T_in).*cells(5,:); % kJ energy

end

