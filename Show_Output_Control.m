function Show_Output_Control(t,p,I,prms,phii1,phii2,tc)

global M_dot_in phi1 cells dt frac_second

figure(2)
drawnow
subplot(221)
plot(t,p);
title(strcat('Air flow : ',num2str(M_dot_in*1000),' g/s, Phi : ',num2str(phi1),' Control Fuel % :',num2str(frac_second*100)));
xlabel('Time')
ylabel('Pressure (Pa)')
fprintf('Avg. Comb. Temp. (at %2.4f seconds): %2.2f K\n',I*dt,mean(cells(5,:)));
subplot(222)
plot(tc,prms);
xlabel('Time')
ylabel('P_{rms}')
subplot(223)
plot(tc,phii1);
xlabel('Time')
ylabel('Primary \phi')
subplot(224)
plot(tc,phii2);
xlabel('Time')
ylabel('Secondary \phi')



end

