% Function for ODE solver for cascading tanks system.

function x_dot=tanks(k1,k2,k3,k4,k5,time,u,t,x)
uint=interp1(time,u,t);
x_dot=zeros(2,1);
x_dot(1)=-k1*sqrt(x(1))+k4*uint;
if x(1)>10
    x_dot(2)=k2*sqrt(x(1))-k3*sqrt(x(2))+k5*uint;
else
    x_dot(2)=k2*sqrt(x(1))-k3*sqrt(x(2));
end