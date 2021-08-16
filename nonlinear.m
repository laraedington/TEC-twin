% Function for ODE solver for nonlinear SDOF system

function y_dot=nonlinear(m,c1,c2,c3,k1,k2,k3,time,F,t,y)
Fint=interp1(time,F,t);
y_dot=zeros(2,1);
y_dot(1)=y(2);
y_dot(2)=(Fint-c1*y(2)-c2*y(2)^2-c3*y(2)^3-k1*y(1)-k2*y(1)^2-k3*y(1)^3)/m;