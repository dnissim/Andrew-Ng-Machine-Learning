%% Clear and Close Figures
clear ; close all; clc

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% Start testing data

figure;
num_iters = 250;
legend_labels=[];
for alpha =[1 0.3 0.1 0.03 0.01 0.003 0.001]
    
    
    % Init Theta and Run Gradient Descent 
    theta = zeros(3, 1);
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    
    % Plot the convergence graph
    
    plot(1:numel(J_history), J_history, 'LineWidth', 1);
    legend_labels=[legend_labels "\alpha="+alpha];
    hold on
    
    fprintf('alpha=')
    fprintf('%g \n',alpha)
    fprintf('Iterations=')
    fprintf('%g \n',length(J_history))
    fprintf('Min Cost=')
    fprintf('%e \n \n',J_history(end))
end
xlabel('Number of iterations');
ylabel('Cost J');
legend(legend_labels)
