clear

pkg load io

# load reference poses
filename = "reference_poses.csv";
C = csv2cell(filename);

ref = cell2struct( C(2:end,2:4).', C(1,2:4)  );
ref = ref;
t = cell2mat(C(2:end,1));

# load longitudinal speeds

filename = "longitudinal_speeds.csv";
C = csv2cell(filename);
v = cell2struct( C(2:end,2:end).', C(1,2:end)  );
v = cell2mat(C(2:end,2));

# load angular velocities

filename = "angular_velocities.csv";
C = csv2cell(filename);
omega = cell2struct( C(2:end,2:end).', C(1,2:end)  );
omega = cell2mat(C(2:end,2));

# load simulated noisy poses and detections
filename = "noisy_poses_simulation_1hz.csv";
C = csv2cell(filename);
filename = "simulation_detections.csv";
C_detections = csv2cell(filename);
#C = cell2mat(C)

i = 2;
k = 2;
for j = 1:length(t)
  # pose
  if i > length(C) ||  t(j) != C{i,1}
    Pose = struct();
    Pose.x = NaN;
    Pose.y = NaN;
    Pose.heading = NaN;
    gnss(end+1,:) = Pose;
  else
    Pose = cell2struct( C(i,2:4).', C(1,2:4)  );
    gnss(end+1,:) = Pose;
    i++;
  endif
  
  # detections
  Detections = struct();
  Detections.x = [];
  Detections.y = [];
  Detections.x_map = [];
  Detections.y_map = [];
  while  k < length(C_detections) && t(j) == C_detections{k,1}
    Detections.x(end+1,:) = C_detections{k,2};
    Detections.y(end+1,:) = C_detections{k,3};
    Detections.x_map(end+1,:) = C_detections{k,4};
    Detections.y_map(end+1,:) = C_detections{k,5};
    k++;
  endwhile
  obs(end+1,:) = Detections;
endfor
t = t*1e-6 - t(1) * 1e-6;
clear C* Detections Pose ans filename i j k l tmp_*

save("-mat7-binary", "data.mat","gnss", "obs", "omega", "ref", "t", "v")