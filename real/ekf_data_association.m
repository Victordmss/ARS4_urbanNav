clear;
close all;

% Load data
load("data.mat");

% PUT DISPLAY = FALSE TO DELETE THE DATA ASSOCIATION PROCESS DISPLAY
display = false;

% Initialize EKF variables
% Initial state: x, y, theta, v, omega
x = [gnss(1).x; gnss(1).y; gnss(1).heading; v(1); omega(1)];
P = diag([0.5, 0.5, 0.1, 0.1, 0.1]); % Initial covariance

% Noise matrices
Q = diag([0.01, 0.01, 0.001, 0.01, 0.001]); % Process noise
R_gnss = diag([0.2, 0.2, 0.01]); % GNSS observation noise
R_lidar = diag([0.1, 0.1]); % Lidar observation noise

% Time step
dt = mean(diff(t));

% Storage for EKF estimates
ekf_estimates = zeros(length(t), 5); % Storing all state variables
ekf_estimates(1, :) = x';
lidar_observations = [];

% EKF loop
for k = 2:length(t)
    % Prediction Step
    % State prediction
    x = [x(1) + x(4) * dt * cos(x(3));
         x(2) + x(4) * dt * sin(x(3));
         x(3) + x(5) * dt;
         v(k);
         omega(k);];

    % Jacobian of the state transition function
    F = [1, 0, -x(4) * dt * sin(x(3)), dt * cos(x(3)), 0;
         0, 1, x(4) * dt * cos(x(3)), dt * sin(x(3)), 0;
         0, 0, 1, 0, dt;
         0, 0, 0, 1, 0;
         0, 0, 0, 0, 1];

    % Covariance prediction
    P = F * P * F' + Q;

    % Correction Step with GNSS
    if ~isnan(gnss(k).x)
        z_gnss = [gnss(k).x; gnss(k).y; gnss(k).heading];
        C_gnss = zeros(3, 5);
        C_gnss(1:3, 1:3) = eye(3); % Observation Jacobian matrix for GNSS
        z_pred_gnss = C_gnss * x; % Predicted GNSS measurement
        K_gnss = P * C_gnss' / (C_gnss * P * C_gnss' + R_gnss); % Kalman gain for GNSS
        x = x + K_gnss * (z_gnss - z_pred_gnss); % State update with GNSS
        P = (eye(size(P)) - K_gnss * C_gnss) * P; % Covariance update for GNSS
    end

    % NN data fusion algo
    n_landmarks = length(poles_obs(k).x);
    x_map = [];
    y_map = [];
    x_lidar_map = [];
    y_lidar_map = [];

    for i = 1:n_landmarks
        % Calculer la position des pôles dans le cadre véhicule (par rapport à la position du robot)
        x_lidar_map(i) = x(1) + cos(x(3)) * poles_obs(k).x(i) - sin(x(3)) * poles_obs(k).y(i);
        y_lidar_map(i) = x(2) + sin(x(3)) * poles_obs(k).x(i) + cos(x(3)) * poles_obs(k).y(i);
    end

    % Pour chaque point LiDAR, calculer la distance par rapport aux points de la carte
    for i = 1:n_landmarks
        % Calculer la distance entre le point LiDAR et tous les points de la carte
        distances = sqrt((map(:, 1) - x_lidar_map(i)).^2 + (map(:, 2) - y_lidar_map(i)).^2);

        % Trouver l'indice du pôle de la carte le plus proche
        [~, idx] = min(distances);

        % Sauvegarder l'association
        x_map(i) = map(idx, 1);  % coordonnées X de la carte associée
        y_map(i) = map(idx, 2);  % coordonnées Y de la carte associée
    end

    % Display
    if display && ~isempty(x_map)
        clf;
        hold on;

        plot(x_lidar_map, y_lidar_map, 'ro', 'DisplayName', 'LiDAR Detections');
        plot(x(1), x(2), 'bo', 'DisplayName', 'Robot Position');
        plot(x_map, y_map, 'go', 'DisplayName', 'Map Pole Position Selected');
        plot([ref.x], [ref.y], 'DisplayName', 'Reference');

        legend;
        title('Association LiDAR Detections with Map Points');
        xlabel('East (m)');
        ylabel('North (m)');
        grid on;
        pause(0.1);
    end


    % Correction Step with Lidar
    if ~isempty(x_map)
        z_lidar = [];
        z_pred_lidar = [];
        C_lidar = zeros(2 * n_landmarks, 5);

        for i = 1:n_landmarks
            r = sqrt(poles_obs(k).x(i)^2 + poles_obs(k).y(i)^2);
            bearing = atan2(poles_obs(k).y(i), poles_obs(k).x(i));


            % ------------------------------------
            % Line to delete LiDAR obs computation
            delta_x = x_map(i) - x(1);
            delta_y = y_map(i) - x(2);
            r = sqrt(delta_x^2 + delta_y^2);
            bearing = atan2(delta_y, delta_x) - x(3);
            % ------------------------------------

            % Lidar measurements in polar coordinates
            z_lidar = [z_lidar; r; bearing];

            % Predicted Lidar measurement
            delta_x = x_map(i) - x(1);
            delta_y = y_map(i) - x(2);
            r_pred = sqrt(delta_x^2 + delta_y^2);
            bearing_pred = atan2(delta_y, delta_x) - x(3);
            z_pred_lidar = [z_pred_lidar; r_pred; bearing_pred];

            % Jacobian matrix for Lidar
            row_idx = 2 * (i - 1) + 1;
            C_lidar(row_idx, 1) = -(delta_x / r_pred);
            C_lidar(row_idx, 2) = -(delta_y / r_pred);
            C_lidar(row_idx, 3) = 0;

            C_lidar(row_idx + 1, 1) = delta_y / (r_pred^2);
            C_lidar(row_idx + 1, 2) = -delta_x / (r_pred^2);
            C_lidar(row_idx + 1, 3) = -1;
        end

        % Kalman gain for Lidar
        R_lidar_full = kron(eye(n_landmarks), R_lidar);
        K_lidar = P * C_lidar' / (C_lidar * P * C_lidar' + R_lidar_full);

        % State update with Lidar
        x = x + K_lidar * (z_lidar - z_pred_lidar);

        % Covariance update for Lidar
        P = (eye(size(P)) - K_lidar * C_lidar) * P;

        % Convert range and bearing to Cartesian for visualization
        for i = 1:n_landmarks
            range = z_lidar(2 * (i - 1) + 1);
            bearing = z_lidar(2 * (i - 1) + 2);
            obs_x = x(1) + range * cos(bearing + x(3));
            obs_y = x(2) + range * sin(bearing + x(3));
        end
    end

    % Save EKF estimate for visualization
    ekf_estimates(k, :) = x';
end

% Plot results
figure;
plot([ref.x], [ref.y], 'g-', 'DisplayName', 'Reference');
hold on;
plot(ekf_estimates(:, 1), ekf_estimates(:, 2), 'b-', 'DisplayName', 'EKF Estimate');
plot([gnss.x], [gnss.y], 'ro', 'DisplayName', 'GNSS Observations');
legend;
title('EKF Localization with GNSS and Lidar Observations');
xlabel('East (m)');
ylabel('North (m)');
grid on;
