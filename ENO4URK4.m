% Domain and discretization
L = pi; % Domain length
N = 200; % Number of grid points
dx = 2*L/N; % Grid spacing
x = linspace(-L + dx/2, L - dx/2, N); % Cell centers
CFL = 1; % CFL number
dt = CFL*dx; % Time step
T = 0.4; % Final time
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
        u1 = dt * compute_flux(u, dx);
        u1 = apply_periodic_bc(u1);

        % Stage 2: Compute u^(2)
        u2 =  dt * compute_flux(u+u1/2, dx);
        u2 = apply_periodic_bc(u2);

        % Stage 3: Compute u^(3)
        u3 = dt * compute_flux(u+u2/2, dx);
        u3 = apply_periodic_bc(u3);

        % Stage 4: Compute u^(4)
        u4 = dt * compute_flux(u+u3, dx);
        u4 = apply_periodic_bc(u4);

        % Stage 5: Compute u^(n+1)
        u = u+u1/6+u2/3+u3/3+u4/6;
        u = apply_periodic_bc(u);
end

% Exact solution 
% uexact = u0(x - T);		% (evaluated at cell centers)
uexact = u0CellAvg(x - T);		% Cell averages

% Plot results
figure;
plot(x,uexact, 'r-',  'LineWidth', 1.5); hold on;
plot(x, RemoveGhosts(u), 'bo', 'LineWidth', 1.5);
xlabel('x');
ylabel('u');
%legend('Exact Solution', 'Finite Volume Solution');
title('4th-Order ENO Reconstruction with Upwind Flux');
grid on;

function v = addGhosts(vInt)
    v=[vInt(end-3:end), vInt, vInt(1:3)];
end

function vInt = RemoveGhosts(v)
    vInt=v(5:end-3);
end

% ENO reconstruction function with stencil selection
function poly = eno_reconstruction(u, i, dx)
    % Periodic boundary conditions
    i_left = i-1; 
    i_left2 = i-2;
    i_left3 = i-3;
    i_right = i+1;
    i_right2 = i+2;
    i_right3 = i+3;

    % Divided differences
    D1_left = (u(i) - u(i_left)) / dx;
    D1_left2 = (u(i_left) - u(i_left2)) / dx;
    D1_left3 = (u(i_left2) - u(i_left3)) / dx;
    D1_right = (u(i_right) - u(i)) / dx;
    D1_right2 = (u(i_right2) - u(i_right)) / dx;
    D1_right3 = (u(i_right3) - u(i_right2)) / dx;
    D2_case1 = (D1_left - D1_left2) / (2*dx);
    D2_case2 = (D1_right - D1_left) / (2*dx);
    D2_case3 = (D1_right2 - D1_right) / (2*dx);
    D2_extral = (D1_left2 - D1_left3) / (2*dx);
    D2_extrar = (D1_right3 - D1_right2) / (2*dx);
    D3_case1 = (D2_case1 - D2_extral) / (3*dx);
    D3_case2 = (D2_case2 - D2_case1) / (3*dx);
    D3_case3 = (D2_case3 - D2_case2) / (3*dx);
    D3_case4 = (D2_extrar - D2_case3) / (3*dx);


        if abs(D1_left) < abs(D1_right)
            if abs(D2_case1) < abs(D2_case2)
                if abs(D3_case1) < abs(D3_case2)
                    % Use stencil {i-3, i-2, i-1, i}
                    poly= (25/12)*u(i)-(23/12)*u(i_left)+(13/12)*u(i_left2)-(1/4)*u(i_left3);
                else
                    poly= (1/4)*u(i_right)+(13/12)*u(i)-(5/12)*u(i_left)+(1/12)*u(i_left2);
                end
            else
                if abs(D3_case2) < abs(D3_case3)
                    % Use stencil {i-2, i-1, i, i+1}
                    poly= (1/4)*u(i_right)+(13/12)*u(i)-(5/12)*u(i_left)+(1/12)*u(i_left2);
                else
                    % Use stencil {i-1, i, i+1, i+2}
                    poly= -(1/12)*u(i_right2)+(7/12)*u(i_right)+(7/12)*u(i)-(1/12)*u(i_left);
                end

            end
        else
            if abs(D2_case2) < abs(D2_case3)
                if abs(D3_case2) < abs(D3_case3)
                    % Use stencil {i-2, i-1, i, i+1}
                    poly= (1/4)*u(i_right)+(13/12)*u(i)-(5/12)*u(i_left)+(1/12)*u(i_left2);
                else
                    % Use stencil {i-1, i, i+1, i+2}
                    poly= -(1/12)*u(i_right2)+(7/12)*u(i_right)+(7/12)*u(i)-(1/12)*u(i_left);
                end
            else
                if abs(D3_case3) < abs(D3_case4)
                    % Use stencil {i-1, i, i+1, i+2}
                    poly= -(1/12)*u(i_right2)+(7/12)*u(i_right)+(7/12)*u(i)-(1/12)*u(i_left);
                else
                    % Use stencil {i, i+1, i+2, i+3}
                    poly= (1/4)*u(i)+(13/12)*u(i_right)-(5/12)*u(i_right2)+(1/12)*u(i_right3);
                end
            end
        end
end

function L_u = compute_flux(u, dx)
    N = length(u); 
    flux = zeros(1, N);

    for i = 5:N-2
        % ENO reconstruction
        uL = eno_reconstruction(u, i-1,dx);

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