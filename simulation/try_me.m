%AR4 mini-project

clear
close all
load("data.mat")

% plot speed
figure;
plot(t,v);
title('Speed')
ylabel('m.s-1 (s)')
xlabel('t (s)')

% plot angular velocity
figure;
plot(t,omega);
title('Angular velocity')
ylabel('rad.s-1 (s)')
xlabel('t (s)')

% ref trajectory, GNSS, obs and map
figure;
plot([ref.x], [ref.y]);
hold on;
plot([gnss.x], [gnss.y],"b*");

for i=1:length(t)
    tmp_obs = obs(i);
    tmp_gnss = gnss(i);

    tmp_obs.x = tmp_gnss.x + obs(i).x * cos(tmp_gnss.heading) - obs(i).y * sin(tmp_gnss.heading);
    tmp_obs.y = tmp_gnss.y + obs(i).x * sin(tmp_gnss.heading) + obs(i).y * cos(tmp_gnss.heading);

    plot(tmp_obs.x, tmp_obs.y, "r.");
    plot(tmp_obs.x_map, tmp_obs.y_map, "k*");
end
legend('Ref', 'GNSS', 'Obs','Map')
ylabel('North (m)')
xlabel('East (m)')

% plot observations dynamically
figure;
for i=1:length(t)
    clf
    plot([ref.x], [ref.y]);
    title(['Lidar observations ',num2str(i),'/',num2str(length(t))]);
    ylabel('North (m)')
    xlabel('East (m)')
    hold on
    %plot([gnss.x], [gnss.y],"m+");
    tmp_obs = obs(i);
    tmp_gnss = gnss(i);
    if isnan(tmp_gnss.x) || isempty(tmp_obs.x)
        continue
    else
        tmp_obs.x = tmp_gnss.x + obs(i).x * cos(tmp_gnss.heading) - obs(i).y * sin(tmp_gnss.heading);
        tmp_obs.y = tmp_gnss.y + obs(i).x * sin(tmp_gnss.heading) + obs(i).y * cos(tmp_gnss.heading);
        for j=1:length(tmp_obs.x)
            plot([tmp_gnss.x,tmp_obs.x(j)], [tmp_gnss.y,tmp_obs.y(j)], "k");
        end
        plot(tmp_obs.x_map, tmp_obs.y_map, "k*");
    end
    pause(0.1);

end

