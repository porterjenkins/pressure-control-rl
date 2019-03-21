close all;
clear all;
clc;

%% List Global Variables

%global dt totalsteps
global dt
totalsteps = 50000;

%% Boundary Conditions

total_time = 30;    % Total time
Mass_in = 30;       % Inlet Mass Flow of Air (g/s)
phi_primary = 1.0;  % Phi of Primary Burner
frac_sec = 0.0;     % Fraction of Primary Burner flow in Second Burner

%% Geometry Variables

pos_primary = 0.20;
pos_secondary = 0.8;
pos_ignition = 0.215;

%% Acoustic Variables

damp_coeff = 0.0; %0.0038;

%% Control Variables

Threshold = 80;
Control_Freq = 20;  % In Hz
rept = 500;      % Reporting Interval
kp = 0.10;

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
        tc(c) = i*dt; % time regime for controller
        %Mass_in = 80;
        %phi_primary = 0.4;
        if i>= 5*cntr_time
            frac_sec = min(kp*max((prms(c) - Threshold),0),1); % Chandra's control - pur RL controller here preferably
        end
        phii1(c) = phi_primary;
        phii2(c) = phii1(c)*frac_sec;
        c = c+1;
    end

    
    %% Show Output
    
    %Show_Output(t,p,i);
    Show_Output_Control(t,p,i,prms,phii1,phii2,tc);
    %Save_GIF(i,rept);
    toc
    tic 
end
disp(p)
toc

