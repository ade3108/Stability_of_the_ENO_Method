% Parameters matching Figure 2d from the paper
L = pi;           % Domain length
N = 200;          % Number of grid points
CFL = 0.75;        % CFL number
dx = 2*L/N;       % Grid spacing
dt = CFL*dx;      % Time step
T_final = 2.1;    % Final time
nt = round(T_final/dt);  % Number of time steps
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers

% Initial condition: u0(x) = sin^4(x)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x)); % Antiderivative
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx; % Cell averages

% Initialize solutions
u_init = u0CellAvg(x);
u_init = addGhosts(u_init);

% Run simulations for both stencils
[L1l, time_points] = RK(x, dt, nt, u_init,u0CellAvg, dx, @eno_reconstructionl);
[L1ml, ~] = RK(x, dt, nt, u_init,u0CellAvg, dx, @eno_reconstructionml);
[L1mr, ~] = RK(x, dt, nt, u_init,u0CellAvg, dx, @eno_reconstructionmr);
[L1r, ~] = RK(x, dt, nt, u_init,u0CellAvg, dx, @eno_reconstructionr);


% Plot results
figure;
semilogy(time_points, L1l, 'b-', 'LineWidth', 1.5); hold on;
semilogy(time_points, L1ml, 'y.-.', 'LineWidth', 1.5); hold on;
semilogy(time_points, L1mr, 'm.-.', 'LineWidth', 1.5); hold on;
semilogy(time_points, L1r, 'g--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('L^1 error');
title('L^1 Error Evolution for Fixed Stencils');
legend('Left-biased stencil','Middle Left-biased stencil', 'Middle Right-biased stencil','Right-biased stencil');
grid off;

function v = addGhosts(vInt)
    v=[vInt(end-3:end), vInt, vInt(1:3)];
end

function vInt = RemoveGhosts(v)
    vInt=v(5:end-3);
end


% Time stepping loop
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

% ENO reconstruction function with stencil selection
function poly = eno_reconstructionl(u, i)
    i_left = i-1; 
    i_left2 = i-2;
    i_left3 = i-3;
    poly= (25/12)*u(i)-(23/12)*u(i_left)+(13/12)*u(i_left2)-(1/4)*u(i_left3); 
end

function poly = eno_reconstructionml(u, i)
    i_left = i-1;
    i_left2 = i-2;
    i_right = i+1;
    poly= (1/4)*u(i_right)+(13/12)*u(i)-(5/12)*u(i_left)+(1/12)*u(i_left2);
end

function poly = eno_reconstructionmr(u, i)
    i_left = i-1;
    i_right = i+1;
    i_right2 = i+2;
    poly= -(1/12)*u(i_right2)+(7/12)*u(i_right)+(7/12)*u(i)-(1/12)*u(i_left);
end

function poly = eno_reconstructionr(u, i)
    i_right = i+1;
    i_right2 = i+2;
    i_right3 = i+3;
    poly= (1/4)*u(i)+(13/12)*u(i_right)-(5/12)*u(i_right2)+(1/12)*u(i_right3);
end

function L_u = compute_flux(u, dx, eno_reconstruction)
    N = length(u); 
    flux = zeros(1, N);

    for i = 5:N-2
        % ENO reconstruction
        uL = eno_reconstruction(u, i-1);

        % Upwind flux
        flux(i) = uL;
    end

    L_u = zeros(1, N);
    for j = 5:N-3
        L_u(j) = -(1/dx) * (flux(j+1) - flux(j));
    end
    L_u = apply_periodic_bc(L_u);
end


% Apply periodic boundary conditions to ghost cells
function u = apply_periodic_bc(u)
    N = length(u) - 7;  % Number of interior cells
    % Left ghost cells
    u(1:4) = u(N+1:N+4);
    % Right ghost cells
    u(N+5:N+7) = u(5:7);
end
