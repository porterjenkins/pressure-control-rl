function Show_Output(t,p,I)

global M_dot_in phi1 cells x dt frac_second

drawnow
subplot(221)
plot(t,p);
title(strcat('Air flow : ',num2str(M_dot_in*1000),' g/s, Phi : ',num2str(phi1),' Control Fuel % :',num2str(frac_second*100)));
xlabel('Time')
ylabel('Pressure (Pa)')
fprintf('Avg. Comb. Temp. (at %2.4f seconds): %2.2f K\n',I*dt,mean(cells(5,:)));
subplot(222)
plot(x,cells(2,:)./cells(3,:));
xlabel('X-Distance')
ylabel('Propane Mass Frac.')
subplot(223)
plot(x,cells(1,:)./cells(3,:));
xlabel('X-Distance')
ylabel('Oxygen Mass Frac.')
subplot(224)
plot(x,cells(5,:));
xlabel('X-Distance')
ylabel('Temperature (K)')

% subplot(231)
% plot(t,p);
% title(strcat('Air flow : ',num2str(M_dot_in*1000),' g/s, Phi : ',num2str(phi1),' Control Fuel % :',num2str(frac_second*100)));
% xlabel('Time')
% ylabel('Pressure')
% fprintf('Avg. Comb. Temp. (at %2.4f seconds): %2.2f K\n',I*dt,mean(cells(5,:)));
% subplot(232)
% plot(x,cells(2,:));
% xlabel('X-Distance')
% ylabel('Propane')
% subplot(233)
% plot(x,cells(1,:));
% xlabel('X-Distance')
% ylabel('Oxygen')
% subplot(234)
% plot(x,cells(5,:));
% xlabel('X-Distance')
% ylabel('Temperature')
% subplot(235)
% plot(x,cells(3,:));
% xlabel('X-Distance')
% ylabel('Total Mass')
% subplot(236)
% plot(x,cells(4,:));
% xlabel('X-Distance')
% ylabel('Heat Content')
% 

end

