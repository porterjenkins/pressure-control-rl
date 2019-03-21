function [p,t] = Time_Solver(steps,I,Mass_in,phi_primary,frac_sec)

% I = i; steps = rept;

global Q_dot_old
global c gamma cells nn L zeta1 N A ig_start dt ig_stop divs ig_pos Aa Ea MW_Fuel rho
global h w X n_Ox_n_F Q_dot Hf T_in Cp_mix U_in M_dot_in n_moles_in_ox MW_Ox phi1
global f1_index f2_index m_dot_Fuel1 m_dot_Fuel2 dFdt Pamb phi dphi kn xp ig_temp frac_second dx

M_dot_in = Mass_in;
M_dot_in = M_dot_in/1000;
phi1 = phi_primary;
frac_second = frac_sec;
U_in = M_dot_in/(rho*h*w);
m_dot_Fuel1 = phi1*M_dot_in*MW_Fuel/(MW_Ox*n_Ox_n_F*n_moles_in_ox);
m_dot_Fuel2 = m_dot_Fuel1*frac_second;

count = 1;

for i = I+1-steps:I
    %% Recompute A Matrix
    
%     c = sqrt(gamma*287*mean(cells(5,:)));
%     omega_n = c*(2*nn-1)*0.5*pi/L;
%     damping = 2*zeta1*omega_n;
%     
%     for k = 1:2*N
%         if rem(k,2) == 1
%             A(k,k+1) = 1;
%         else
%             A(k,k-1) = -omega_n(k/2)*omega_n(k/2);
%             A(k,k) = -damping(k/2);
%         end
%     end
%     
%     %% RK for acoustics
%     
%     l1 = dt*(A*X);
%     l2 = dt*(A*(X + l1/2));
%     l3 = dt*(A*(X + l2/2));
%     l4 = dt*(A*(X + l3));
%     
%     X = X + (l1 + 2*l2 +2*l3 + l4)/6;
%     
%     B = zeros(2*N,1);
    
    %% Compute Combustion
    if i >= ceil(ig_start/dt) && i <= ceil(ig_stop/dt)
        cells(5,ceil(ig_pos/L*divs)) = max(cells(5,ceil(ig_pos/L*divs)),ig_temp);
        cells(4,ceil(ig_pos/L*divs)) = cells(3,ceil(ig_pos/L*divs))*Cp_mix(cells(5,ceil(ig_pos/L*divs)))*cells(5,ceil(ig_pos/L*divs));
    end
    
    Q_dot_old = Q_dot;
    
    for j = 1:divs
        ddFdt = dFdt(Aa,Ea,cells(5,j),...
            (cells(2,j)/MW_Fuel)/(L/divs*h*w),...
            (cells(1,j)/32)/(L/divs*h*w)); % kmol/(m^3-s)
        cells(2,j) = cells(2,j) - ddFdt*dt*(L/divs*h*w)*MW_Fuel;
        cells(1,j) = cells(1,j) - n_Ox_n_F*ddFdt*dt*(L/divs*h*w)*32;
        
        if cells(2,j)<0 | cells(1,j)<0
            ddFdt1 = min((cells(2,j) + ddFdt*dt*(L/divs*h*w)*MW_Fuel)/(dt*MW_Fuel*(L/divs*h*w)),...
                (cells(1,j) + n_Ox_n_F*ddFdt*dt*(L/divs*h*w)*32)/(32*n_Ox_n_F*dt*(L/divs*h*w)));
            cells(2,j) = cells(2,j) + (ddFdt - ddFdt1)*dt*(L/divs*h*w)*MW_Fuel;
            cells(1,j) = cells(1,j) + n_Ox_n_F*(ddFdt - ddFdt1)*dt*(L/divs*h*w)*32;
            ddFdt = ddFdt1;
        end
        
        Q_dot(j) =  - ddFdt*Hf*1000 - 0*10.45*2*(L/divs)*(w+h)*(cells(5,j) - T_in)/(1000*(L/divs*h*w));
%         Q_max = 0.2e5;
%         if max(Q_dot)>Q_max
%             Q_dot(find(Q_dot>Q_max)) = Q_max;
%         end
        cells(4,j) = cells(4,j) + Q_dot(j)*(L/divs*h*w)*dt;
        cells(5,j) = cells(4,j)/(Cp_mix(cells(5,j))*cells(3,j));
    end    
    
    %% Find and add Exciting Input
    
        c = sqrt(gamma*287*mean(cells(5,:)));
    omega_n = c*(2*nn-1)*0.5*pi/L;
    damping = 2*zeta1*omega_n;
    
    for k = 1:2*N
        if rem(k,2) == 1
            A(k,k+1) = 1;
        else
            A(k,k-1) = -omega_n(k/2)*omega_n(k/2);
            A(k,k) = -damping(k/2);
        end
    end
    
    %% RK for acoustics
    
    l1 = dt*(A*X);
    l2 = dt*(A*(X + l1/2));
    l3 = dt*(A*(X + l2/2));
    l4 = dt*(A*(X + l3));
    
    X = X + (l1 + 2*l2 +2*l3 + l4)/6;
    
    B = zeros(2*N,1);
    
    for j = 1:divs
        B(2:2:2*N) = B(2:2:2*N) + (((gamma - 1))/Pamb*phi((j-1)*L/divs)*(Q_dot(j) - Q_dot_old(j))*1000/dt)'*dx/L;
    end
    
    X = X + B*dt;
    
    clear l1 l2 l3 l4
    
    %% Solve Spatial Shifting
    
    cells_temp = zeros(5,divs);
    
    u_max = -10000;
    
    for j = 1:divs
        orand = randn;
        u_prime = (X(2:2:2*N)'./(kn.*kn))*dphi((j-1)*L/divs)'/gamma;
%         u_prime = sign((X(2:2:2*N)'./(kn.*kn))*dphi((j-1)*L/divs)'/gamma)...
%             *min(abs((X(2:2:2*N)'./(kn.*kn))*dphi((j-1)*L/divs)'/gamma),2*U_in*cells(5,j)/T_in);
        
        if j<=ceil(ig_pos/L*divs)+1
            divs_net = ((u_prime + (1+0.02*orand)*U_in)*dt)/(L/divs);
        else
            u_max = max(u_max,U_in*cells(5,j)/T_in);
            divs_net = ((u_prime + (1+0.02*orand)*u_max)*dt)/(L/divs);
        end
        m_divs_net = abs(divs_net);
        
        if u_max>=dx/dt
            fprintf('\nWARNING! POSSIBLE DIVERGENCE!!\n');
        end
        
        if divs_net>=0
            shift = ceil(divs_net);
        else
            shift = floor(divs_net);
        end
    
        if abs(shift) == 1
            per(1) = m_divs_net;
        else
            %disp(divs)
            %disp(dt)
            %disp(divs_net)
            %disp(shift)
            for k = 1:abs(shift)-1
                %disp(k)
                per(k) = 1/m_divs_net;
            end
            per(k+1) = (m_divs_net - floor(m_divs_net))/m_divs_net;
        end
        for k = 1:abs(shift)
            if (j+sign(shift)*k<=0 | j+sign(shift)*k>divs)~=1
                cells_temp(:,j+sign(shift)*k) = cells(:,j)*per(k);
            end
        end
        
        cells(:,j) = cells(:,j)*(1-min(1,m_divs_net));
    end
    
    cells = cells + cells_temp;
    
    for j = 1:divs
        cells(5,j) = cells(4,j)/(cells(3,j)*Cp_mix(cells(5,j)));
    end
    
    rnd = randn;
    cells(1,1) = cells(1,1) + (1 + 0.01*rnd)*M_dot_in*dt*32/(n_moles_in_ox*MW_Ox);
    cells(3,1) = cells(3,1) + (1 + 0.01*rnd)*M_dot_in*dt;
    cells(5,1) = T_in;
    cells(4,1) = cells(3,1)*Cp_mix(cells(5,1)).*cells(5,1);
    
    cells(3,f1_index) = cells(3,f1_index) - cells(2,f1_index);
    cells(2,f1_index) = cells(2,f1_index) + (1 + 0.005*randn)*m_dot_Fuel1*dt;
    cells(3,f1_index) = cells(3,f1_index) + cells(2,f1_index);
    cells(4,f1_index) = cells(3,f1_index)*Cp_mix(cells(5,f1_index)).*cells(5,f1_index);
    
    cells(3,f2_index) = cells(3,f2_index) - cells(2,f2_index);
    cells(2,f2_index) = cells(2,f2_index) + (1 + 0.005*randn)*m_dot_Fuel2*dt;
    cells(3,f2_index) = cells(3,f2_index) + cells(2,f2_index);
    cells(4,f2_index) = cells(3,f2_index)*Cp_mix(cells(5,f2_index)).*cells(5,f2_index);
    
    if ((u_prime + (1+0.02*orand)*u_max)*dt)/(L/divs)<0
        divs_net = ((u_prime + (1+0.02*orand)*u_max)*dt)/(L/divs);
        shift = floor(divs_net);
        
        for j = -1:-1:shift
            cells(:,divs+j+1) = cells(:,divs+shift);
        end
    end
    
    p(count) = Pamb*phi(xp)*X(1:2:2*N);
    t(count) = i*dt; 
    
    count = count+1;
   
end

end

