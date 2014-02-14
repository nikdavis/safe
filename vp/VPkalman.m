%function kalman(duration, dt)
% function kalman(duration, dt)
%
% Kalman filter simulation for a vehicle travelling along a road.
% INPUTS
%   duration = length of simulation (seconds)
%   dt = step size (seconds)

data = csvread('vp.csv');
frame = data(:, 1);
duration = size(frame);
dt = 0.015;
% axis = 2 for y 
% axis = 3 for x
axis = 3;
meas = data(:, axis);

measnoise = 30;     % position measurement noise (feet)
accelnoise = 200;   % acceleration noise (feet/sec^2)

a = [1 dt; 0 1];    % transition matrix
b = [dt^2/2; dt];   % input matrix
c = [1 0];          % measurement matrix

if axis == 2
    xinit = 400;
elseif axis == 3
    xinit = 200;
end
x = [xinit; 0];         % initial state vector
xhat = x;           % initial state estimate

Sz = measnoise^2;   % measurement error covariance
Sw = accelnoise^2 * [dt^4/4 dt^3/2; dt^3/2 dt^2]; % process noise cov
P = Sw;             % initial estimation covariance

% Initialize arrays for later plotting.
pos = [];       % true position array
poshat = [];    % estimated position array
posmeas = [];   % measured position array
vel = [];       % true velocity array
velhat = [];    % estimated velocity array

if axis == 2
    truncate = 800;
elseif axis == 3
    truncate = 500;
end

%for t = 0 : dt: (duration(1) - 1)*dt,
for n = 1 : duration(1)
    % Use a constant commanded acceleration of 1 foot/sec^2.
    u = 0;
    % Simulate the linear system.
    ProcessNoise = accelnoise * [(dt^2/2)*randn; dt*randn]; % nhieu theo gia toc => nhieu vi tri va van toc
    %desire_x = a * x + b * u;
    x = a * x + b * u + ProcessNoise;
    % Extrapolate the most recent state estimate to the present time.
    xhat = a * xhat + b * u;
    % Simulate the noisy measurement
    %MeasNoise = measnoise * randn;
    %y = c * x + MeasNoise;
    y = meas(n);
    if ((y > 0) && (y < truncate))
        % Form the Innovation vector.
        Inn = y - c * xhat;
        % Compute the covariance of the Innovation.
        s = c * P * c' + Sz;
        % Form the Kalman Gain matrix.
        K = a * P * c' * inv(s);
        % Update the state estimate.
        xhat = xhat + K * Inn;
        % Compute the covariance of the estimation error.
        P = a * P * a' - a * P * c' * inv(s) * c * P * a' + Sw;
    else
        y = -1;
    end
    % Save some parameters for plotting later.
    pos = [pos; x(1)];
    posmeas = [posmeas; y];
    poshat = [poshat; xhat(1)];
    vel = [vel; x(2)];
    velhat = [velhat; xhat(2)];
end
% Plot the results
close all;
t = 0 : dt: (duration(1) - 1)*dt

%figure(1)
figure(1);
plot(t, posmeas, t, poshat);
legend('pos meas', 'estimation pos error');
grid;
xlabel('Time (sec)');
ylabel('Position (feet)');
title('Figure 1 - Vehicle Position (True, Measured, and Estimated)')

% figure(2)
% figure(2);
% plot(t,pos-posmeas, t,pos-poshat);
% legend('Position Measurement Error','Position Estimation Error ');
% grid;
% xlabel('Time (sec)');
% ylabel('Position Error (feet)');
% title('Figure 2 - Position Measurement Error and Position Estimation Error');

figure(3)
figure(3);
plot(t,vel, t,velhat);
legend('Velocity True','Velocity estimated ');
grid;
xlabel('Time (sec)');
ylabel('Velocity (feet/sec)');
title('Figure 3 - Velocity (True and Estimated)');

figure(4)
figure(4);
plot(t,vel-velhat);
legend('Velocity Estimation Error');
grid;
xlabel('Time (sec)');
ylabel('Velocity Error (feet/sec)');
title('Figure 4 - Velocity Estimation Error');
