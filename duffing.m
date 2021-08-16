% Function for ODE solver for SDOF Duffing which allows for changing k3

function y_dot=duffing(m,c1,k1,k3,time,F,t,y)
Fint=interp1(time,F,t);
k3int=interp1(time,k3,t);
y_dot=zeros(2,1);
y_dot(1)=y(2);
y_dot(2)=(Fint-c1*y(2)-k1*y(1)-k3int*y(1)^3)/m;