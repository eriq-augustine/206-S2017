function [x, objectiveValue, g, xHistory, objectiveHistory] = mymin5(initialX)
   Q = [2, 1; 1, 2];
   b = [10; 10];

   % If the initial values will overflow, then just take the log of them.
   if (initialX' * initialX == Inf || exp(initialX(1)) == Inf || exp(initialX(2)) == Inf)
      initialX = [log(initialX(1)); log(initialX(2))];
   end;

   x = initialX;
   xHistory = initialX;                        % xHistory will store all the x's we visit

   objectiveValue = .5 * x' * Q * x - b' * x + exp(x(1) / 10) + exp(2 * x(2) / 10);
   objectiveHistory = objectiveValue;                      % objectiveHistory array stores all fvals we see
   oldObjectiveValue = Inf;  % Since we are just starting, best objective found in past step is infinity

   while abs(objectiveValue - oldObjectiveValue) > 1e-12
      H = hessian(x, Q);              % Compute Hessian to use for next n = 2 steps
                                      % (This is the 'Q' of the quadratic approximation)
      g_center = gradient(x, Q, b)';  % Compute gradient at the center of the quadratic approximation
      b_lin = H * x - g_center;       % This finds the 'b' term of the quadratic approximation

      for j = 1:2                     % Loop through n = 2 steps
         g = H * x - b_lin;           % Compute gradient from quadratic approximation
         if j == 1
               d = -g;                % First step should be a gradient step
         else
               gamma = (g' * H * d) / (d' * H * d);
               d = -(g - gamma * d);
         end;

         % If gradient miniscule, avoid dividing by something close to zero
         if norm(g) > 1e-9
            % Choose step size to minimize quadratic approximation:
            % alpha = -g' * d / (d' * H * d);

            % Uncomment below to choose step size using Newton's method
            alpha = linesearch(x, d, Q, b);
         else
            alpha = 0;
         end;

         x = x + alpha * d;  % Update x
         oldObjectiveValue = objectiveValue;
         objectiveValue = .5 * x' * Q * x - b' * x + exp(x(1) / 10) + exp(2 * x(2) / 10);
         xHistory = [xHistory, x];  % Log this x in array to keep track of our path
         objectiveHistory = [objectiveHistory, objectiveValue];  % Log this objectiveValue in array to keep track of fvals seen
      end;
   end;

   g = gradient(x, Q, b)';

function alpha = linesearch(x, d, Q, b)
   % This is a Newton's method line search
   alpha = 0;
   for j = 1:100
      alpha = alpha + -gradient(x + alpha * d, Q, b) * d  /  (d' * hessian(x + alpha * d, Q) * d);
   end;

function grad = gradient(x, Q, b)
   grad = x' * Q  - b' + (1 / 10) * [exp(x(1) / 10) , 2 * exp(2 * x(2) / 10) ];

function H = hessian(x, Q)
   H = Q + 1 / (100) * [exp(x(1) / 10), 0; 0, 4 * exp(2 * x(2) / 10)];
