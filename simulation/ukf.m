clear;
close all;

% Load data
load("data.mat");

function [xPts, wPts, nPts] = SigmaPoints_cholesky(x,P,w0)
%Symmetric sampling 2n+1 sigma points
n=length(x);
nPts=2*n+1;

%The sigma points are placed one after another in a matrix
xPts = zeros(n,2*n+1); %memory allocation

M=chol(P,'lower');%Cholesky factorization
scale=sqrt(n/(1-w0));
for i=1:n
    xPts(:,i)=x+scale*M(:,i);
    xPts(:,i+n)=x-scale*M(:,i);
end

%We add the average point at the end
xPts(:,nPts)=x;

%Weights
wPts=ones(1,nPts)*(1-w0)/(2*n);
wPts(nPts)=w0;
end





% 1 - Initialization
x = [ref(1).x; ref(1).y; ref(1).heading; v(1); omega(1)];      % Initial state
P = diag([0.01, 0.01, 0.1, 0.1, 0.1]);                            % Initial covariance
n = 5;         % State dimension
N = 2*n+1;     % 2n+1 sampling

% Noise matrices
Q = diag([1, 1, 0.1, 0.1, 0.1]);       % Process noise
R = diag([0.2, 0.2, 0.01, 0.1, 0.1]);        % Observation noise

% Time step
dt = mean(diff(t));

% Storage for UKF estimates
ukf_estimates = zeros(length(t), 3);         % Only store x, y, theta for visualization
ukf_estimates(1, :) = x(1:3)';

% 2 - Sampling loop
for k = 2:length(t)
    n_landmarks = length(obs(k).x);
    m = 3 + 2 * n_landmarks;  % observation dimension

    % 2.1 - Prediction step

    % Generate sigma points
    [xPts, wPts, nPts] = SigmaPoints_cholesky(x, P, N);  % Generate sigma points

    % Propagate the sigma points through the observation model
    yPts = zeros(m, nPts); % Matrix of predicted sigma points
    for i = 1:nPts
      yPts_temp = [];

      for j = 1:n_landmarks
          % Compute the LiDAR observation
          x_lidar = cos(xPts(3, i)) * (obs(k).x_map(j) - xPts(1, i)) ...
                    + sin(xPts(3, i)) * (obs(k).y_map(j) - xPts(2, i));
          y_lidar = -sin(xPts(3, i)) * (obs(k).x_map(j) - xPts(1, i)) ...
                    + cos(xPts(3, i)) * (obs(k).y_map(j) - xPts(2, i));

          % Add the value to the observation
          yPts_temp = [yPts_temp; obs(k).x(j); obs(k).y(j)];
      end

      % Construction de yPts pour ce point sigma (GNSS + Lidar)
      yPts(:, i) = [
          xPts(1, i);          % x_GNSS
          xPts(2, i);          % y_GNSS
          xPts(3, i);          % theta_GNSS
          yPts_temp;           % LiDAR obs
      ];
    end

    % Compute predicted observation moments
    % 1. Prediction of the observation (=mean)
    y_pred = yPts * wPts';
    % 2. Covariance matrix of the observation
    R = blkdiag(diag([0.2, 0.2, 0.01]), kron(eye(n_landmarks), diag([0.1, 0.1])));
    P_y = R;
    for i = 1:nPts
        P_y = P_y + wPts(i) * (yPts(:,i) - y_pred) * (yPts(:,i) - y_pred)';
    end

    % Compute Cross Covariance matrix
    P_xy = zeros(n, m);  % Cross-covariance matrix between state and measurement
    for i = 1:nPts
        P_xy = P_xy + wPts(i) * (xPts(:,i) - x) * (yPts(:,i) - y_pred)';
    end

    # Get measures (GNSS)
    z_gnss = [x(1); x(2); x(3)];
    if ~isnan(gnss(k).x)
        fprintf("Mesure GNSS\n");
        % Measurement
        z_gnss = [gnss(k).x; gnss(k).y; gnss(k).heading;];
    end

    # Get measures (LiDAR)
    z_lidar = [];
    if ~isempty(obs(k))
        fprintf("Mesure LiDAR\n");

        for idx = 1:n_landmarks
            z_lidar = [z_lidar; obs(k).x(idx); obs(k).y(idx)];
        end

    end

    # Create the observation vector
    z = [z_gnss; z_lidar];

    % Update kalman gain estimation
    K = P_xy * inv(P_y);

    % Update state estimate
    innovation = z - y_pred
    x = x + K * innovation;

    % Update covariance estimate
    P = P - K * P_y * K';

    % Save UKF estimate for visualization
    ukf_estimates(k, :) = x(1:3)';

    % 2.2 Prediction for the next step
    [xPts, wPts, nPts] = SigmaPoints_cholesky(x, P, N);  % Generate sigma points
    for i = 1:nPts
      xPts(:, i) = [
        xPts(1,i) + xPts(4,i) * dt * cos(xPts(3,i));
        xPts(2,i) + xPts(4,i) * dt * sin(xPts(3,i));
        xPts(3,i) + xPts(5,i) * dt;
        v(k);
        omega(k);
    ];
    end

    x = xPts * wPts';
    P = Q;
    for i = 1:nPts
        P = P + wPts(i) * (xPts(:,i) - x) * (xPts(:,i) - x)';
    end
end


% Plot results
figure;
plot([ref.x], [ref.y], 'g-', 'DisplayName', 'Reference');
hold on;
plot(ukf_estimates(:, 1), ukf_estimates(:, 2), 'b-', 'DisplayName', 'UKF Estimate');
plot([gnss.x], [gnss.y], 'ro', 'DisplayName', 'GNSS Observations');
plot(lidar_observations(:, 1), lidar_observations(:, 2), 'kx', 'DisplayName', 'Lidar Observations');
legend;
title('UKF Localization with GNSS and Lidar Observations');
xlabel('East (m)');
ylabel('North (m)');
grid on;
