clear;
close all;

pkg load statistics;

% Load data
load("data.mat");

% Initialize EKF variables
% Initial state: x, y, theta, v, omega
x = [gnss(1).x; gnss(1).y; gnss(1).heading; v(1); omega(1)];
P = diag([0.1, 0.1, 0.1, 0.5, 0.5]); % Initial covariance

% Noise matrices
Q = diag([0.1, 0.1, 0.7, 0.1, 0.5]); % Process noise
R_gnss = diag([0.1, 0.1, 0.01]); % GNSS observation noise
R_lidar = diag([0.7, 0.7]); % Lidar observation noise

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

    % Correction Step with Lidar
    if ~isempty(obs(k).x_map)
        n_landmarks = length(obs(k).x_map);
        z_lidar = [];
        z_pred_lidar = [];
        C_lidar = zeros(2 * n_landmarks, 5);

        for i = 1:n_landmarks
            r = sqrt(obs(k).x(i)^2 + obs(k).y(i)^2);
            bearing = atan2(obs(k).y(i), obs(k).x(i));

            % Lidar measurements in polar coordinates
            z_lidar = [z_lidar; r; bearing];

            % Predicted Lidar measurement
            delta_x = obs(k).x_map(i) - x(1);
            delta_y = obs(k).y_map(i) - x(2);
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
h = plot([ref.x], [ref.y], 'g-', 'DisplayName', 'Reference');
hold on;
plot(ekf_estimates(:, 1), ekf_estimates(:, 2), 'b-', 'DisplayName', 'EKF Estimate');
plot([gnss.x], [gnss.y], 'ro', 'DisplayName', 'GNSS Observations');
legend;
title('EKF Localization with GNSS and Lidar Observations');
xlabel('East (m)');
ylabel('North (m)');
grid on;

waitfor(h)


Px1 = P(1,1);
Px2 = P(2,2);
Px3 = P(3,3);

%Metrics
ex=ekf_estimates(:, 1)'-[ref.x];
ex = ex';
ex_m=mean(ex);
ex_abs_max=max(abs(ex));
ex_MSE=mean((ex-ex_m).^2);
%Find the percentile for the chi-square distribution for any DoF
DoF=1; %degree of freedom 
Pr=0.99; %Chosen percentile
Th=chi2inv(Pr,DoF); %Threshold
Consistency_x=mean(ex./Px1'.*ex< Th);
disp(['Mean Error in x= ', num2str(ex_m)]);
disp(['Max Error in x= ', num2str(ex_abs_max)]);
disp(['Mean Square Error in x= ', num2str(mean(ex_MSE))]);
disp(['Consistency in x= ', num2str(Consistency_x)]);
disp('');

ex=ekf_estimates(:, 2)'-[ref.y];
ex = ex';
ex_m=mean(ex);
ex_abs_max=max(abs(ex));
ex_MSE=mean((ex-ex_m).^2);
%Find the percentile for the chi-square distribution for any DoF
DoF=1; %degree of freedom 
Pr=0.99; %Chosen percentile
Th=chi2inv(Pr,DoF); %Threshold
Consistency_y=mean((ex./Px2'.*ex)< Th);
disp(['Mean Error in y= ', num2str(ex_m)]);
disp(['Max Error in y= ', num2str(ex_abs_max)]);
disp(['Mean Square Error in y= ', num2str(mean(ex_MSE))]);
disp(['Consistency in y= ', num2str(Consistency_y)]);
disp('');

ex=ekf_estimates(:, 3)'-[ref.heading];
ex = ex';
ex_m=mean(ex);
ex_abs_max=max(abs(ex));
ex_MSE=mean((ex-ex_m).^2);
%Find the percentile for the chi-square distribution for any DoF
DoF=1; %degree of freedom 
Pr=0.99; %Chosen percentile
Th=chi2inv(Pr,DoF); %Threshold
Consistency_h=mean((ex./Px3'.*ex)< Th);
disp(['Mean Error in h= ', num2str(ex_m)]);
disp(['Max Error in h= ', num2str(ex_abs_max)]);
disp(['Mean Square Error in h= ', num2str(mean(ex_MSE))]);
disp(['Consistency in h= ', num2str(Consistency_h)]);
disp('');
