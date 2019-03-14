function Save_GIF(i,rept)

global M_dot_in phi1 m_dot_Fuel2 m_dot_Fuel1
global name

frame = getframe(gcf);
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);

if i == rept
    name = strcat('F',num2str(M_dot_in*1000),'Pi',num2str(floor(phi1*100)),'Pc',num2str(floor(m_dot_Fuel2/m_dot_Fuel1*100)),'.gif');
    imwrite(imind,cm,name,'gif', 'Loopcount',inf,'DelayTime',0.01);
else
    imwrite(imind,cm,name,'gif','WriteMode','append','DelayTime',0.01);
end

end

