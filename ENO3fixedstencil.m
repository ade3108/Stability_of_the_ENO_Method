% Domain and discretization
L = pi; % Domain length
N = 200; % Number of grid points
dx = 2*L/N; % Grid spacing
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers
CFL = 0.75; % CFL number
dt = CFL*dx; % Time step
T = 3.4; % Final time
nt = round(T / dt); % Number of time steps

%u0 = @(x) sin(x).^4; % Initial condition (cell averages, including ghost cells)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x));	% Antiderivative of sin^4
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx;

% u = u0(x);
u = u0CellAvg(x);

u=addGhosts(u);

% Runge-Kutta time stepping

[ul,u_l] = RK(dt, nt, u, dx, @eno_reconstructionl);
[um,u_m] = RK(dt, nt, u, dx, @eno_reconstructionm);
[ur,u_r] = RK(dt, nt, u, dx, @eno_reconstructionr);

% Exact solution (evaluated at cell centers)
uexact = u0CellAvg(x - T);

% Plot results
figure;
plot(ul,'b-', 'LineWidth', 1.5); hold on;
plot(um,'g-', 'LineWidth', 1.5); hold on;
plot(ur,'r--', 'LineWidth', 1.5);
ylabel('Energy');
legend('Left Stencil', 'Middle Stencil','Right Stencil');
title('Energy of fixed stencil');
% plot(x, uexact, 'r-', 'LineWidth', 1.5); hold on;
% plot(x, RemoveGhosts(u_l), 'b-', 'LineWidth', 1.5); hold on;
% plot(x, RemoveGhosts(u_m), 'm-', 'LineWidth', 1.5); hold on;
% plot(x, RemoveGhosts(u_r), 'g-', 'LineWidth', 1.5);
% xlabel('x');
% ylabel('u');
% legend('Exact Solution','Left Stencil','Midddle Stencil', 'Right Stencil');
% title('Fixed Stencil approx');
grid on;


function v = addGhosts(vInt)
    v=[vInt(end-2:end), vInt, vInt(1:2)];
end

function vInt = RemoveGhosts(v)
    vInt=v(4:end-2);
end

% Time stepping loop
function [energy, u] = RK(dt, nt, u, dx, stencil)
    energy = zeros(1,nt);
    for t = 1:nt
        % Stage 1: Compute u^(1)
        u1 = u + dt * compute_flux(u, dx, stencil);
        u = apply_periodic_bc(u1);

        % % Stage 2: Compute u^(2)
        % u2 = (3/4) * u + (1/4) * u1 + (1/4) * dt * compute_flux(u1, dx,stencil);
        % u2 = apply_periodic_bc(u2);
        % 
        % % Stage 3: Compute u^(n+1)
        % u = (1/3) * u + (2/3) * u2 + (2/3) * dt * compute_flux(u2, dx, stencil);
        % u = apply_periodic_bc(u);

        energy(t) = sum(RemoveGhosts(u).^2)*dx;
    end
end

% ENO reconstruction function with stencil selection
function poly = eno_reconstructionl(u, i)
    poly= (11/6)*u(i)-(7/6)*u(i-1)+(1/3)*u(i-2); 
end

function poly = eno_reconstructionm(u, i)
    poly= (1/3)*u(i+1)+(5/6)*u(i)-(1/6)*u(i-1);
end

function poly = eno_reconstructionr(u, i)
    poly= -(1/6)*u(i+2)+(5/6)*u(i+1)+(1/3)*u(i);
end

function L_u = compute_flux(u, dx, eno_reconstruction)
    N = length(u); 
    flux = zeros(1, N);

    for i = 4:N-1
        % ENO reconstruction
        uL = eno_reconstruction(u, i-1);

        % Upwind flux
        flux(i) = uL;
    end

    L_u = zeros(1, N);
    for j = 4:N-2
        L_u(j) = -(1/dx) * (flux(j+1) - flux(j));
    end
    L_u = apply_periodic_bc(L_u);
end


% Apply periodic boundary conditions to ghost cells
function u = apply_periodic_bc(u)
    N = length(u) - 5;  % Number of interior cells
    % Left ghost cells
    u(1:3) = u(N+1:N+3);
    % Right ghost cells
    u(N+4:N+5) = u(4:5);
end
