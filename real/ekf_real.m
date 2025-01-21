clear;
close all;

% Load data
load("data.mat");
display = false;

function F = compute_F_jacobian(x_t, dt)
    % Inputs:
    % x_t: Current state vector [x_t, y_t, theta_t, v_t, omega_t]
    % dt: Time step (delta t)

    % Extract state variables
    theta = x_t(3);
    v = x_t(4);

    % Initialize the Jacobian matrix F
    F = eye(5); % Identity matrix of size 5x5

    % Fill in the non-zero derivatives
    F(1, 3) = -v * dt * sin(theta); % d(x_next)/d(theta)
    F(1, 4) = dt * cos(theta);      % d(x_next)/d(v)

    F(2, 3) = v * dt * cos(theta);  % d(y_next)/d(theta)
    F(2, 4) = dt * sin(theta);      % d(y_next)/d(v)

    F(3, 5) = dt;                   % d(theta_next)/d(omega)
end


% 1 - Initialization
x = [ref(1).x; ref(1).y; ref(1).heading; v(1); omega(1)];       % Initial state
P = diag([0.5, 0.5, 0.1, 0.1, 0.1]);                            % Initial covariance
n = 5;                                                          % State dimension
x_map = [];
y_map = [];

% Noise matrices
Q = diag([0.2, 0.2, 0.01, 0.01, 0.01]);                         % Process noise
n_landmarks_0 = length(poles_obs(1).x);
R = blkdiag(diag([0.2, 0.2, 0.01]), kron(eye(n_landmarks_0), diag([0.1, 0.1])));


% Initialize Jacobian matrix
m = 3 + 2 * n_landmarks_0; % Observation dimension
C = zeros(m, 5);

% GNSS measurements (first 3 rows correspond to x, y, theta)
C(1:3, 1:3) = eye(3);

obs = zeros(n_landmarks_0, 2);
% NN data fusion algo
for i = 1:n_landmarks_0
    % Extract the poles position in vehicule frame
    x_lidar_map = x(1) + cos(x(3))*poles_obs(1).x(i) - sin(x(3))*poles_obs(1).y(i);
    y_lidar_map = x(2) + sin(x(3))*poles_obs(1).x(i) + cos(x(3))*poles_obs(1).y(i);

    % Compute the distance with all the map poles
    distances = sqrt((map(:, 1) - x_lidar_map).^2 + (map(:, 2) - y_lidar_map).^2);

    % Find the closest map index
    [~, idx] = min(distances);

    % Enregistrer l'association (indice détection, indice point carte)
    x_map(i) = map(i, 1);
    y_map(i) = map(i, 2);
end

% LiDAR measurements (one row per landmark, starting from the 4th row)
for i = 1:n_landmarks_0
    % Compute delta values for the current landmark
    delta_x = x_map(i) - x(1);
    delta_y = y_map(i) - x(2);

    % Fill jacobian values
    row_idx = 3 + (i - 1) * 2;
    C(row_idx, 1) = -cos(x(3));
    C(row_idx, 2) = -sin(x(3));
    C(row_idx, 3) = -sin(x(3)) * delta_x + cos(x(3)) * delta_y;

    C(row_idx + 1, 1) = sin(x(3));
    C(row_idx + 1, 2) = -cos(x(3));
    C(row_idx + 1, 3) = -cos(x(3)) * delta_x - sin(x(3)) * delta_y;
end


K = P * C' / (C * P * C' + R); % Initial kalman gain

% Time step
dt = mean(diff(t));

% Storage for EKF estimates
ekf_estimates = zeros(length(t), 3);         % Only store x, y, theta for visualization
ekf_estimates(1, :) = x(1:3)';


% 2 - Loop over time steps
for k = 2:length(t)
    n_landmarks = length(poles_obs(k).x);
    m = 3 + n_landmarks * 2; % Observation dimension

    % Réinitialisation des variables à chaque itération
    x_map = [];
    y_map = [];
    x_lidar_map = [];
    y_lidar_map = [];
    obs = zeros(n_landmarks, 2);

    % NN data fusion algo
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

    % Affichage sur la même figure
    if display && ~isempty(x_map)
        clf; % Effacer la figure précédente
        hold on; % Maintenir l'affichage précédent

        % Affichage des détections LiDAR
        plot(x_lidar_map, y_lidar_map, 'ro', 'DisplayName', 'LiDAR Detections');

        % Affichage de la position du robot
        plot(x(1), x(2), 'bo', 'DisplayName', 'Robot Position');

        % Affichage des pôles de la carte sélectionnés
        plot(x_map, y_map, 'go', 'DisplayName', 'Map Pole Position Selected');

        % Affichage de la référence
        plot([ref.x], [ref.y], 'v-', 'DisplayName', 'Reference');

        % Ajouter des légendes
        legend;
        title('Association LiDAR Detections with Map Points');
        xlabel('East (m)');
        ylabel('North (m)');
        grid on;

        % Mettre à jour la figure
        pause(0.1); % Ajuster la durée selon la vitesse désirée
    end



    % 2.1 - Get measures
    % Get measurements (LiDAR)
    z_lidar = [];
    y_pred_lidar = [];
    if ~isempty(poles_obs(k))
        fprintf("LiDAR measure \n");
        for idx = 1:n_landmarks
            z_lidar = [z_lidar; poles_obs(k).x(idx); poles_obs(k).y(idx)];
            x_lidar = cos(x(3))*(x_map(idx) - x(1)) + sin(x(3)) * (y_map(idx) - x(2));
            y_lidar = -sin(x(3))*(x_map(idx) - x(1)) + cos(x(3)) * (y_map(idx) - x(2));
            y_pred_lidar = [y_pred_lidar; poles_obs(k).x(idx); poles_obs(k).y(idx)];    % Avoid lidar OBS
        end
    end

    % Get measurements (GNSS)

    z_gnss = [x(1); x(2); x(3)];
    y_pred_GNSS = [x(1); x(2); x(3)];
    if ~isnan(gnss(k).x)
        fprintf("GNSS measure\n");
        z_gnss = [gnss(k).x; gnss(k).y; gnss(k).heading;];
    end

    % Create the observation vector
    z = [z_gnss; z_lidar];

    % 2.2 Estimation step

    y_pred = [y_pred_GNSS ; y_pred_lidar];

    % Observation jacobian computation
    C = zeros(m, 5);

    % GNSS measurements (first 3 rows correspond to x, y, theta)
    C(1:3, 1:3) = eye(3);

    % LiDAR measurements (one row per landmark, starting from the 4th row)
    for i = 1:n_landmarks
        % Compute delta values for the current landmark
        delta_x = x_map(i) - x(1);
        delta_y = y_map(i) - x(2);

        % Fill jacobian values
        row_idx = 3 + (i - 1) * 2;
        C(row_idx, 1) = -cos(x(3));
        C(row_idx, 2) = -sin(x(3));
        C(row_idx, 3) = -sin(x(3)) * delta_x + cos(x(3)) * delta_y;

        C(row_idx + 1, 1) = sin(x(3));
        C(row_idx + 1, 2) = -cos(x(3));
        C(row_idx + 1, 3) = -cos(x(3)) * delta_x - sin(x(3)) * delta_y;
    end

    R = blkdiag(diag([0.2, 0.2, 0.01]), kron(eye(n_landmarks), diag([0.1, 0.1])));

    % Predict K
    K = P * C' * pinv(C * P * C' + R);
    innovation = z - y_pred;
    x = x + K * innovation;

    P = (eye(n) - K*C) * P * (eye(n) - K*C)' + K * R * K';
    % 2.3 Save EKF estimate for visualization
    ekf_estimates(k, :) = x(1:3)';

    % 2.4 Prediction step
    A = compute_F_jacobian(x, dt);

    % Predict state
    x = [
        x(1) + x(4) * dt * cos(x(3));
        x(2) + x(4) * dt * sin(x(3));
        x(3) + x(5) * dt;
        v(k);
        omega(k);
    ];

    % Predict covariance
    P = A * P * A' + Q;

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

