% Parameters matching Figure 2d from the paper
L = pi;           % Domain length
N = 200;          % Number of grid points
CFL = 0.75;        % CFL number
dx = 2*L/N;       % Grid spacing
dt = CFL*dx;      % Time step
T_final = 5;    % Final time
nt = round(T_final/dt);  % Number of time steps
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers

% Initial condition: u0(x) = sin^4(x)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x)); % Antiderivative
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx; % Cell averages

% Initialize solutions
u_init = u0CellAvg(x);
u_init = addGhosts(u_init);

% Run simulations for both stencils
[L1_left, time_points] = RK(x, dt, nt, u_init,u0CellAvg, dx, @left_biased_stencil);
[L1_right, ~] = RK(x, dt, nt, u_init,u0CellAvg, dx, @right_biased_stencil);

% Plot results
figure;
semilogy(time_points, L1_left, 'b-', 'LineWidth', 1.5); hold on;
semilogy(time_points, L1_right, 'g-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('L^1 error');
title('L^1 Error Evolution for Fixed Stencils');
legend('Left-biased stencil', 'Right-biased stencil');
grid off;

%% Helper functions
function v = addGhosts(vInt)
    v = [vInt(end-1:end), vInt, vInt(1)]; % 2 ghost cells each side
end

function vInt = RemoveGhosts(v)
    vInt = v(3:end-1); % Remove ghost cells
end

function [L1_error, time_points] = RK(x, dt, nt, u,u0CellAvg, dx, stencil_func)
    L1_error = zeros(1, nt);
    time_points = zeros(1, nt);
    
    for t = 1:nt
        current_time = t*dt;
        time_points(t) = current_time;
        
        % Forward Euler step
        u = u + dt * compute_flux(u, dx, stencil_func);
        u = apply_periodic_bc(u);
        
        % Compute exact solution and L1 error
        uexact = u0CellAvg(x - current_time);
        L1_error(t) = dx*sum(abs(RemoveGhosts(u) - uexact));
    end
end

% Second-order stencil functions
function poly = left_biased_stencil(u, i)
    % Left-biased second-order reconstruction
    poly = -(1/2)*u(i-1) + (3/2)*u(i);
end

function poly = right_biased_stencil(u, i)
    % Right-biased second-order reconstruction
    poly = (1/2)*u(i) + (1/2)*u(i+1);
end

function L_u = compute_flux(u, dx, stencil_func)
    N = length(u); 
    flux = zeros(1, N);

    for i = 3:N
        % Use the specified stencil
        uL = stencil_func(u, i-1);
        flux(i) = uL; % Upwind flux
    end

    L_u = zeros(1, N);
    for j = 3:N-1
        L_u(j) = -(1/dx) * (flux(j+1) - flux(j));
    end
    L_u = apply_periodic_bc(L_u);
end

function u = apply_periodic_bc(u)
    N = length(u) - 3;  % Number of interior cells
    u(1:2) = u(N+1:N+2); % Left ghost cells
    u(N+3) = u(3);       % Right ghost cell
end
