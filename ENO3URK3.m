% Domain and discretization
L = pi; % Domain length
N = 200; % Number of grid points
dx = 2*L/N; % Grid spacing
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers
CFL = 0.5; % CFL number
dt = CFL*dx; % Time step
T = 0.8; % Final time
nt = round(T / dt); % Number of time steps

%u0 = @(x) sin(x).^4; % Initial condition (cell averages, including ghost cells)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x));	% Antiderivative of sin^4
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx;

% u = u0(x);
u = u0CellAvg(x);

u=addGhosts(u);

% Time stepping loop
for t = 1:nt
        % Stage 1: Compute u^(1)
        u1 = u + dt * compute_flux(u, dx);
        u1 = apply_periodic_bc(u1);

        % Stage 2: Compute u^(2)
        u2 = (3/4) * u + (1/4) * u1 + (1/4) * dt * compute_flux(u1, dx);
        u2 = apply_periodic_bc(u2);

        % Stage 3: Compute u^(n+1)
        u = (1/3) * u + (2/3) * u2 + (2/3) * dt * compute_flux(u2, dx);
        u = apply_periodic_bc(u);
end

% Exact solution (evaluated at cell centers)
uexact = u0CellAvg(x - T);		% Cell averages

% Plot results
figure;
plot(x,uexact, 'r-',  'LineWidth', 1.5); hold on;
plot(x, RemoveGhosts(u), 'bo', 'LineWidth', 1.5);
xlabel('x');
ylabel('u');
%legend('Exact Solution', 'Finite Volume Solution');
title('3rd-Order ENO Reconstruction with Upwind Flux');
grid on;

function v = addGhosts(vInt)
    v=[vInt(end-2:end), vInt, vInt(1:2)];
end

function vInt = RemoveGhosts(v)
    vInt=v(4:end-2);
end

% ENO reconstruction function with stencil selection
function poly = eno_reconstruction(u, i, dx)
    % Periodic boundary conditions
    i_left2 = i-2;
    i_left = i-1;
    i_right = i+1;
    i_right2 = i+2;

    % First-order divided differences

    D1_left = (u(i) - u(i_left)) / dx;
    D1_left2 = (u(i_left) - u(i_left2)) / dx;
    D1_right = (u(i_right) - u(i)) / dx;
    D1_right2 = (u(i_right2) - u(i_right)) / dx;
    D2_case1 = (D1_left - D1_left2) / (2*dx);
    D2_case2 = (D1_right - D1_left) / (2*dx);
    D2_case3 = (D1_right2 - D1_right) / (2*dx);

    if abs(D1_left) < abs(D1_right)
        if abs(D2_case1) < abs(D2_case2)
            poly= (11/6)*u(i)-(7/6)*u(i_left)+(1/3)*u(i_left2);
        else
            poly= (1/3)*u(i_right)+(5/6)*u(i)-(1/6)*u(i_left);
        end
    else
        if abs(D2_case2) < abs(D2_case3)
            poly= (1/3)*u(i_right)+(5/6)*u(i)-(1/6)*u(i_left);
        else
            poly= -(1/6)*u(i_right2)+(5/6)*u(i_right)+(1/3)*u(i);
        end
    end
end



function L_u = compute_flux(u, dx)
    N = length(u); 
    flux = zeros(1, N);

    for i = 4:N-1
        % ENO reconstruction
        uL = eno_reconstruction(u, i-1,dx);

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
