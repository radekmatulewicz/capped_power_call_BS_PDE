function [S, w, oPrice, delta, oDelta] = CappedPowerCallPDE(S0, r, sigma, T, K, c, i, Mt, Mx, smin, smax, theta)
% Black Scholes zamiana zmiennych
% na rownanie ciepla. 

% Zamiana zmiennych
tauMax = sigma.^2 * T / 2;  % tauMax = sigma^2*T/2 when t=0          
dtau = tauMax / Mt;         % Time steps
tau = 0:dtau:tauMax;        % Discretization of time 

xMax = log(smax/K);         % xMax = ln(smax/K)
if(smin <= 0)
    xMin = log(eps/K);      % xMin = ln(smin/K)
else
    xMin = log(smin/K);     % xMin = ln(smin/K)
end
dx = (xMax - xMin)/Mx;      % Price steps
x = xMin + 0:dx:xMax;       % Discretization of price    
   
q = r/(sigma.^2/2);         % q = 2r/sigma^2

lambda = dtau/(dx*dx);      % lambda = Dtau / (Dx)^2
%fprintf('Lambda = %f \n', lambda);
% Definicja macierzy trojdiagonalnych
alphab(1:Mx-1, 1) = 1. + 2.*theta.*lambda;
betab(1:Mx-1, 1) = -theta.*lambda; 
B = spdiags([betab alphab betab], -1:1, Mx-1, Mx-1);

alphac(1:Mx-1, 1) =  1. - 2.*(1. - theta).*lambda;
betac(1:Mx-1, 1) = (1. - theta).*lambda;
C = spdiags([betac alphac betac], -1:1, Mx-1, Mx-1);

% Definiujemy wektor z warunkami brzegowymi:
d = zeros(size(C,2),1);

w = zeros(Mx+1, 1);

% Warunek poczatkowy
w(2:Mx) = exp(0.5 .* x(2:Mx) .* (q-1)) .* min(max(K^(i-1) .* exp(i.* x(2:Mx)) - 1, 0) , c/K);

for k = 1:Mt
     pom = zeros(Mx+1,1);
     
     % warunek brzegowy x -> -inf
     w(1) = 0;  
     pom(1) = 0;
     
     % warunek brzegowy x -> +inf
     w(end) = exp(0.5 * (q-1) * xMax + (0.25 * (q+1).^2)*tau(k)) * c/K * exp(-2*r*tau(k) /(sigma*sigma));
     pom(end) = exp(0.5 * (q-1) * xMax + (0.25 * (q+1).^2)*tau(k+1)) * c/K * exp(-2*r*tau(k+1) /(sigma*sigma));
     
     d(1) = -betab(1).*pom(1) + betac(1).*w(1);
     d(end) = -betab(1).*pom(end) + betac(1).*w(end);
     
     % Rozwiazujemy uklad rownan
     pom(2:Mx) = B \ (C*w(2:Mx) + d);
        
     w = pom;
end

% Wracamy do zmiennych
S = K*exp(x);
w = K*exp(-0.5*(q-1)*x' - (0.25*(q-1)^2 + q)*tauMax).*w;

% Delta:
delta = zeros(Mx+1, 1);
for k = 2:Mx
    delta(k) = (w(k+1) - w(k-1)) / (2*(S(k+1) - S(k-1)));
end

delta = delta ./ S0;

% Interpolujemy w S0
oPrice = interp1(S, w ,S0);
oDelta = interp1(S, delta, S0);

end
