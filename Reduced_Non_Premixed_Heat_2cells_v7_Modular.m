close all;
clear all;
clc;

%% List Global Variables

global dt totalsteps

%% Boundary Conditions

total_time = 10;    % Total time
Mass_in = 40;       % Inlet Mass Flow of Air (g/s)
phi_primary = 1.0;  % Phi of Primary Burner
frac_sec = 0.0;     % Fraction of Primary Burner flow in Second Burner

%% Geometry Variables

pos_primary = 0.3;
pos_secondary = 0.6;
pos_ignition = 0.31;

%% Acoustic Variables

damp_coeff = 0.003; %0.0038;

%% Setup all Values

Setup_BC(total_time,Mass_in,phi_primary,frac_sec);
Setup_Geometry(pos_primary,pos_secondary,pos_ignition);
Setup_Chemistry()
Setup_Acoustic(damp_coeff)

%% Internal Initialization

Initialize_Solution()

%% Clear Setup Variables

clear total_time pos_primary pos_secondary damp_coeff pos_ignition

%%

Control_Freq = 20;  % In Hz

%% Temporal Evolution

rept = 100;      % Reporting Interval
cntr_time = 1/(dt*Control_Freq);       % Control Interval
p = [];
t = [];

tic

for i = rept:rept:totalsteps
    
    % Solve over the given reporting time
    
    [p1,t1] = Time_Solver(rept,i,Mass_in,phi_primary,frac_sec);
    
%     if i*dt > 2
%         phi_primary = 0.8;
%     end
    
    p = [p,p1];
    t = [t,t1];
    
    clear p1 p2
    
    %% Show Output
    
    Show_Output(t,p,i);
    %Save_GIF(i,rept);
    toc
    tic 
end
toc