function  Save_Image()


global M_dot_in phi1 m_dot_Fuel2 m_dot_Fuel1
global name

frame = getframe(gcf);
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);

name = strcat('F',num2str(M_dot_in*1000),'Pi',num2str(floor(phi1*100)),'Pc',num2str(floor(m_dot_Fuel2/m_dot_Fuel1*100)),'.png');
imwrite(imind,cm,name,'png');


end

