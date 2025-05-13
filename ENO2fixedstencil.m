% Domain and discretization
L = pi; % Domain length
N = 200; % Number of grid points
dx = 2*L/N; % Grid spacing
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers
CFL = 0.75; % CFL number
dt = CFL*dx; % Time step
T = 0.4; % Final time
nt = round(T / dt); % Number of time steps

%u0 = @(x) sin(x).^4; % Initial condition (cell averages, including ghost cells)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x));	% Antiderivative of sin^4
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx;

% u = u0(x);
u = u0CellAvg(x);

u=addGhosts(u);

% Runge-Kutta time stepping
[u_l,ul] = RK(dt, nt, u, dx, @eno_reconstructionl);
[u_r,ur] = RK(dt, nt, u, dx, @eno_reconstructionr);



% Exact solution (evaluated at cell centers)
uexact = u0CellAvg(x - T);		% Cell averages


% Plot results
figure;
plot(u_l,'b-', 'LineWidth', 1.5); hold on;
plot(u_r,'g--', 'LineWidth', 1.5);
ylabel('Energy');
legend('Left Stencil', 'Right Stencil');
title('Energy of fixed stencil');
% plot(x, uexact, 'r-', 'LineWidth', 1.5); hold on;
% plot(x, RemoveGhosts(ul), 'b-', 'LineWidth', 1.5); hold on;
% plot(x, RemoveGhosts(ur), 'g-', 'LineWidth', 1.5);
% xlabel('x');
% ylabel('u');
% legend('Exact Solution', 'Left Stencil', 'Right Stencil');
% title('Fixed Stencil approx');
grid on;

function v = addGhosts(vInt)
v=[vInt(end-1:end), vInt, vInt(1)];
end

function vInt = RemoveGhosts(v)
    vInt=v(3:end-1);
end


% Time stepping loop
function [energy, u] = RK(dt, nt, u, dx, stencil)
    energy = zeros(1,nt);
    for t = 1:nt
        u1 = u + dt * compute_flux(u, dx, stencil);
        u = apply_periodic_bc(u1);

        % % Stage 1: Compute u^(1)
        % u1 = u + dt * compute_flux(u, dx, stencil);
        % u1 = apply_periodic_bc(u1);
        % 
        % % Stage 2: Compute u^(2)
        % u2 = (3/4) * u + (1/4) * u1 + (1/4) * dt * compute_flux(u1, dx,stencil);
        % u2 = apply_periodic_bc(u2);
        % 
        % % Stage 3: Compute u^(n+1)
        % u = (1/3) * u + (2/3) * u2 + (2/3) * dt * compute_flux(u2, dx, stencil);
        % u = apply_periodic_bc(u);

        energy(t) = sum(RemoveGhosts(u).^2)*dx;
    end
    % figure;
    % plot(energy);
end

% ENO reconstruction function with stencil selection
function poly = eno_reconstructionl(u, i)
    % i is the index including ghost nodes (3:N+2 corresponds to interior)
    poly = -(1/2)*u(i-1) + (3/2)*u(i);
end

function poly = eno_reconstructionr(u, i)
    % i is the index including ghost nodes (3:N+2 corresponds to interior)
    poly = (1/2)*u(i) + (1/2)*u(i+1);
end

function L_u = compute_flux(u, dx, eno_reconstruction)
    N = length(u); 
    flux = zeros(1, N);

    for i = 3:N
        % ENO reconstruction
        uL = eno_reconstruction(u, i-1);

        % Upwind flux
        flux(i) = uL;
    end

    L_u = zeros(1, N);
    for j = 3:N-1
        L_u(j) = -(1/dx) * (flux(j+1) - flux(j));
    end
    L_u=apply_periodic_bc(L_u);
end

% Apply periodic boundary conditions to ghost cells
function u = apply_periodic_bc(u)
    N = length(u) - 3;  % Number of interior cells
    % Left ghost cells
    u(1:2) = u(N+1:N+2);
    % Right ghost cells
    u(N+3) = u(3);
end
