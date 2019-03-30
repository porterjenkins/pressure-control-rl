close all;
clear all;
clc;

%% List Global Variables

global dt totalsteps

%% Boundary Conditions

total_time = 30;    % Total time
Mass_in = 140;       % Inlet Mass Flow of Air (g/s)
phi_primary = 1.0;  % Phi of Primary Burner
frac_sec = 0.0;     % Fraction of Primary Burner flow in Second Burner

%% Geometry Variables

pos_primary = 0.20;
pos_secondary = 0.75;  % MESS
pos_ignition = 0.215;

%% Acoustic Variables

damp_coeff = 0.0008;

%% Control Variables

Threshold = 100;
Control_Freq = 20;  % In Hz
rept = 500;      % Reporting Interval
kp = 1e-3;

%% Setup all Values

Setup_BC(total_time,Mass_in,phi_primary,frac_sec);
Setup_Geometry(pos_primary,pos_secondary,pos_ignition);
Setup_Chemistry()
Setup_Acoustic(damp_coeff)

%% Internal Initialization

Initialize_Solution()

%% Clear Setup Variables

clear total_time pos_primary pos_secondary damp_coeff pos_ignition

%% Temporal Evolution

p = [];
t = [];
phii1 = [];
phii2 = [];
prms = [];
tc = [];
cntr_time = 1/(dt*Control_Freq);       % Control Interval

total_phi = 1;
phi_primary = total_phi;

tic

c = 1;

for i = rept:rept:totalsteps
    
    % Solve over the given reporting time
    
    [p1,t1] = Time_Solver(rept,i,Mass_in,phi_primary,frac_sec);
    
    p = [p,p1]; % pressure
    t = [t,t1]; % time
    
    clear p1 t1
    
    if rem(i,cntr_time) == 0
        prms(c) = rms(p(i-cntr_time+1:i));
        disp(prms)
        tc(c) = i*dt; % time regime for controller
        %Mass_in = 80;
        %phi_primary = 0.4;
        if i>= 5*cntr_time
            frac_sec = min(kp*max((prms(c) - Threshold),0),1); % Chandra's control - put RL controller here preferably
        end
%         phi_primary  = total_phi*(1-frac_sec);
        phii1(c) = phi_primary;
        phii2(c) = total_phi*frac_sec;
        c = c+1;
    end
    
    

    
    %% Show Output
    
    %Show_Output(t,p,i);
    %Show_Output_Control(t, p, i, prms, phii1, phii2, tc);
    %Save_GIF(i,rept);
%     if tc(end) == 5
%        
%         p_title = ['p_', num2str(kp), '.mat'];
%         prms_title = ['prms_', num2str(kp), '.mat'];
%         tc_title = ['tc_', num2str(kp), '.mat'];
%         phii1_title = ['phii1_', num2str(kp), '.mat'];
%         phii2_title = ['phii2_', num2str(kp), '.mat'];
%         
%         save(p_title, 'p')
%         save(prms_title, 'prms')
%         save(tc_title, 'tc')
%         save(phii1_title, 'phii1')
%         save(phii2_title, 'phii2')
%         
%         fig_title = ['figs_', num2str(kp), '.mat'];
%         
%         saveas(gcf, fig_title)
%         
%     end
    toc
    tic 
end
toc