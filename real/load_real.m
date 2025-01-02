
clear

pkg load io

% load reference poses
filename = "reference_poses.csv";
C = csv2cell(filename);

ref = cell2struct( C(2:end,2:4).', C(1,2:4)  ); % x,y, theta
t = cell2mat(C(2:end,1));


% load longitudinal speeds

filename = "longitudinal_speeds.csv";
C = csv2cell(filename);
v= cell2struct( C(2:end,2:end).', C(1,2:end)  );
v = cell2mat(C(2:end,2));

% load angular velocities

filename = "angular_velocities.csv";
C = csv2cell(filename);
omega = cell2struct( C(2:end,2:end).', C(1,2:end)  );
omega = cell2mat(C(2:end,2));

% load poses and detections
filename = "septentrio_poses.csv";
C = csv2cell(filename);
filename = "lidar_poles.csv";
C_detections = csv2cell(filename);
filename = "lidar_signs.csv";
C_signs = csv2cell(filename);
%C = cell2mat(C)


i = 2;
k = 2;
l = 2;

for j = 1:length(t)
  % pose
  if i > length(C) || t(j) != C{i,1}
    Pose = struct();
    tmp_pose.x = NaN;
    tmp_pose.y = NaN;
    tmp_pose.heading = NaN;
    tmp_pose.("varX") = NaN;
    tmp_pose.("varY") = NaN;
    tmp_pose.("varHeading") = NaN;
    gnss(end+1,:) = tmp_pose;
  else
    tmp_pose = cell2struct( C(i,2:end).', C(1,2:end)  );
    gnss(end+1,:) = tmp_pose;
    i++;
  endif
  
  % detection poles
  Detections = struct();
  tmp_detections.x = [];
  tmp_detections.y = [];
  while  k < length(C_detections) && t(j) == C_detections{k,1}
    tmp_detections.x(end+1,:) = C_detections{k,2};
    tmp_detections.y(end+1,:) = C_detections{k,3};
    k++;
  endwhile
  poles_obs(end+1,:) = tmp_detections;
  
  % detection signs
  tmp_detections = struct();
  tmp_detections.x = [];
  tmp_detections.y = [];
  while  l < length(C_signs) && t(j) == C_signs{l,1}
    tmp_detections.x(end+1,:) = C_signs{l,2};
    tmp_detections.y(end+1,:) = C_signs{l,3};
    l++;
  endwhile
  signs_obs(end+1,:) = tmp_detections;
endfor

filename = "map.csv";
map = csv2cell(filename);
map = cell2mat(map(2:end,:));

clear C* Detections Pose ans filename i j k l tmp_*
%C_detections C_signs
t = t*1e-6 - t(1) * 1e-6;
X = cell2mat(struct2cell(ref));
plot(X(1,:), X(2,:));
%plot(M(:,1),M(:,2),"+");

save("-mat7-binary", "data.mat","gnss", "poles_obs", "signs_obs", "map", "omega", "ref", "t", "v")