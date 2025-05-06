% Parameters matching Figure 2d from the paper
L = pi;           % Domain length
N = 200;          % Number of grid points (from paper)
CFL = 0.1;        % CFL number (from paper)
dx = 2*L/N;       % Grid spacing
dt = CFL*dx;      % Time step
T_final = 0.99;    % Final time (long enough to capture error evolution)
nt = round(T_final/dt);  % Number of time steps
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers

% Initial condition: u0(x) = sin^4(x)
U0 = @(x) 1/32*(12*x - 8*sin(2*x) + sin(4*x)); % Antiderivative
u0CellAvg = @(x) (U0(x+dx/2) - U0(x-dx/2))/dx; % Cell averages
u = u0CellAvg(x); % Initial condition
u = addGhosts(u); % Add ghost cells

% Arrays to store error evolution
time_points = zeros(1, nt);
L1_errors = zeros(1, nt);

% Time stepping loop
for t = 1:nt
    current_time = t*dt;
    time_points(t) = current_time;
    
    % RK4 stages
    u1 = dt * compute_flux(u, dx);
    u1 = apply_periodic_bc(u1);
    
    u2 = dt * compute_flux(u + u1/2, dx);
    u2 = apply_periodic_bc(u2);
    
    u3 = dt * compute_flux(u + u2/2, dx);
    u3 = apply_periodic_bc(u3);
    
    u4 = dt * compute_flux(u + u3, dx);
    u4 = apply_periodic_bc(u4);
    
    u = u + u1/6 + u2/3 + u3/3 + u4/6;
    u = apply_periodic_bc(u);
    
    % Compute exact solution and L2 error
    uexact = u0CellAvg(x - current_time);
    L1_errors(t) = dx*sum(abs(uexact - RemoveGhosts(u)));

end

% Plot results matching Figure 2d
figure;
semilogy(time_points, L1_errors, 'b-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('L^1 error');
title('L^1 Error Evolution');
grid on;

function v = addGhosts(vInt)
    v = [vInt(end-3:end), vInt, vInt(1:3)]; % 4 ghost cells each side
end

function vInt = RemoveGhosts(v)
    vInt = v(5:end-3); % Remove ghost cells
end

function L_u = compute_flux(u, dx)
    N = length(u); 
    flux = zeros(1, N);

    % Compute fluxes at cell interfaces
    for i = 5:N-2
        uL = eno_reconstruction(u, i-1, dx); % Left state reconstruction
        flux(i) = uL; % Upwind flux
    end

    % Compute divergence of flux
    L_u = zeros(1, N);
    for j = 5:N-3
        L_u(j) = -(1/dx) * (flux(j+1) - flux(j));
    end
    L_u = apply_periodic_bc(L_u);
end

function u = apply_periodic_bc(u)
    N = length(u) - 7;  % Interior points
    u(1:4) = u(N+1:N+4); % Left BC
    u(N+5:end) = u(5:7); % Right BC
end

function poly = eno_reconstruction(u, i, dx)
    % Calculate divided differences
    D1_left = (u(i) - u(i-1)) / dx;
    D1_left2 = (u(i-1) - u(i-2)) / dx;
    D1_left3 = (u(i-2) - u(i-3)) / dx;
    D1_right = (u(i+1) - u(i)) / dx;
    D1_right2 = (u(i+2) - u(i+1)) / dx;
    D1_right3 = (u(i+3) - u(i+2)) / dx;
    
    D2_case1 = (D1_left - D1_left2) / (2*dx);
    D2_case2 = (D1_right - D1_left) / (2*dx);
    D2_case3 = (D1_right2 - D1_right) / (2*dx);
    D2_extral = (D1_left2 - D1_left3) / (2*dx);
    D2_extrar = (D1_right3 - D1_right2) / (2*dx);
    
    D3_case1 = (D2_case1 - D2_extral) / (3*dx);
    D3_case2 = (D2_case2 - D2_case1) / (3*dx);
    D3_case3 = (D2_case3 - D2_case2) / (3*dx);
    D3_case4 = (D2_extrar - D2_case3) / (3*dx);

    % Stencil selection logic
    if abs(D1_left) < abs(D1_right)
        if abs(D2_case1) < abs(D2_case2)
            if abs(D3_case1) < abs(D3_case2)
                % Stencil {i-3,i-2,i-1,i}
                poly = (25/12)*u(i)-(23/12)*u(i-1)+(13/12)*u(i-2)-(1/4)*u(i-3);
            else
                % Stencil {i-2,i-1,i,i+1}
                poly = (1/4)*u(i+1)+(13/12)*u(i)-(5/12)*u(i-1)+(1/12)*u(i-2);
            end
        else
            if abs(D3_case2) < abs(D3_case3)
                % Stencil {i-2,i-1,i,i+1}
                poly = (1/4)*u(i+1)+(13/12)*u(i)-(5/12)*u(i-1)+(1/12)*u(i-2);
            else
                % Stencil {i-1,i,i+1,i+2}
                poly = -(1/12)*u(i+2)+(7/12)*u(i+1)+(7/12)*u(i)-(1/12)*u(i-1);
            end
        end
    else
        if abs(D2_case2) < abs(D2_case3)
            if abs(D3_case2) < abs(D3_case3)
                % Stencil {i-2,i-1,i,i+1}
                poly = (1/4)*u(i+1)+(13/12)*u(i)-(5/12)*u(i-1)+(1/12)*u(i-2);
            else
                % Stencil {i-1,i,i+1,i+2}
                poly = -(1/12)*u(i+2)+(7/12)*u(i+1)+(7/12)*u(i)-(1/12)*u(i-1);
            end
        else
            if abs(D3_case3) < abs(D3_case4)
                % Stencil {i-1,i,i+1,i+2}
                poly = -(1/12)*u(i+2)+(7/12)*u(i+1)+(7/12)*u(i)-(1/12)*u(i-1);
            else
                % Stencil {i,i+1,i+2,i+3}
                poly = (1/4)*u(i)+(13/12)*u(i+1)-(5/12)*u(i+2)+(1/12)*u(i+3);
            end
        end
    end
end