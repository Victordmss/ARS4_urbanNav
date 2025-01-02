%ARS4
%Real data
clear
close all
load("data.mat")

figure;
plot(t,v);
title('Speed')
ylabel('m.s-1 (s)')
xlabel('t (s)')

figure;
plot(t,omega);
title('Angular velocity')
ylabel('rad.s-1 (s)')
xlabel('t (s)')

%ref trajectory
figure
plot([ref.x], [ref.y])
hold on
plot([gnss.x], [gnss.y],"r+")

legend('Ref', 'GNSS')
ylabel('North (m)')
xlabel('East (m)')

