%%Section1

%%Input the data
%insert the data document
filename = 'DatiProject.xlsx';

%convert to table format
data = readtable("DatiProject.xlsx", 'Sheet', 'Data');
datafactor = readtable("DatiProject.xlsx", 'Sheet', 'DataForFactors');

%display first three of the table
disp(head(data))

%generate the time vector
dates=data.Time;
%%Transform the data
%generate real gdp=GDPCI_t
GDPt= data.GDPC1;

%generate yt = log(GDPC1_t)
yt = log(data.GDPC1);

% generate Δyt = (y_t - y_t-1)
dyt = [NaN; diff(yt)];

% generate logPCECTPI_t
logPCECTPI = log(data.PCECTPI);

% generate π_t = log(PCECTPI_t) - log(PCECTPI_t-1)
pi_t = [NaN; diff(logPCECTPI)];

%generate T spread_t = GS10_t - TB3MS_t
T_spread = data.GS10 - data.TB3MS;


%%Plot all the requested series
% plot yt
subplot(3,2,1);
plot(dates, yt, 'LineWidth', 1.5);
title('yt = log(GDPC1)');
xlabel('Date'); ylabel('yt');
grid on;

% plot Δyt
subplot(3,2,2);
plot(dates, dyt, 'LineWidth', 1.5);
title('\Delta y_t = y_t - y_{t-1}');
xlabel('Date'); ylabel('\Delta y_t');
grid on;

% plot logPCECTPI_t
subplot(3,2,3);
plot(dates, logPCECTPI, 'LineWidth', 1.5);
title('log(PCECTPI_t)');
xlabel('Date'); ylabel('log(PCECTPI_t)');
grid on;

% plot π_t
subplot(3,2,4);
plot(dates, pi_t, 'LineWidth', 1.5);
title('\pi_t = log(PCECTPI_t) - log(PCECTPI_{t-1})');
xlabel('Date'); ylabel('\pi_t');
grid on;

% plot T spread_t
subplot(3,2,5);
plot(dates, T_spread, 'LineWidth', 1.5);
title('T spread_t = GS10_t - TB3MS_t');
xlabel('Date'); ylabel('T spread_t');
grid on;

% adjust the graph
sgtitle('Economic Data Analysis');

%%Section2（preparation for models）
%% Combine processed data into a table
sample_data = table(dates,GDPt, yt, dyt, pi_t, T_spread, ...
               'VariableNames', {'Date','GDPt','yt' ,'dyt', 'pi_t', 'T_spread'});

%%Set the index value
%Beginning of the forecast
idx1 = find(year(sample_data.Date) <=1985 & quarter(sample_data.Date) <2, 1, 'last');
%End of the forecast
idx2= find(year(sample_data.Date) <=2018 & quarter(sample_data.Date) <=3, 1, 'last');

%%Show the sample size
N = size(dates,1); 
disp(['Sample size: ' num2str(N)])
disp(['Sample size 1st estimation sample: ' num2str(sum(idx1))])
disp(['Sample size last estimation sample: ' num2str(sum(idx2))])

%Date range for later plotting
date_range = sample_data.Date(idx1:idx2); 

%% Plot Autocorrelation Function (ACF) of three series in Var(4)
initial_training_set = table(dyt(1:idx1),pi_t(1:idx1), T_spread(1:idx1), ...
               'VariableNames', {'dyt', 'pit', 'T_spread'});
variable_names = {'dyt', 'pi_t', 'T_spread'};
% Loop through each variables and plot ACF and PACF
for i = 1:3
    figure;

    % ACF
    subplot(2, 1, 1);
    autocorr(initial_training_set(:, i), 'NumLags', 20);
    title(['ACF of three series: ', variable_names{i}]);

    % PACF
    subplot(2, 1, 2);
    parcorr(initial_training_set(:, i), 'NumLags', 20);
    title(['PACF of three series: ', variable_names{i}]);
end


% forecast horizon
Hmax =1;
%Number of predictions for later model
num_predictions = idx2 - idx1 ;

%initial value to transfer the diff back to original value(1985,Q1,lnGDP)
initial_value=9.0087;
%generate the actual values series for later calculation
actual_values_GDPt =GDPt(idx1+1:idx2);
actual_values_yt =yt(idx1+1:idx2);
actual_values_dyt =dyt(idx1+1:idx2);
actual_values_pit =pi_t(idx1+1:idx2);
actual_values_Tt =T_spread(idx1+1:idx2);



%%Section 3：Models
%%Model 1: Random walk

%Length of the rolling window
window_size=sum(idx1);

% Initialize the arrays
predictions_rw = NaN(num_predictions, 1);

for i = 1:num_predictions
    % training window
    train_data = yt(idx1 + i - window_size : idx1 + i - 1);
    
    % RW without drift(dyt=dyt-1
    predictions_rw(i) = train_data(end);     
end

% Exponential of the predictions
exp_predictions_rw = exp(predictions_rw);

%% Compute RMSE
rmse_rw1 = sqrt(nanmean((actual_values_GDPt  - exp_predictions_rw).^2));
fprintf('RMSE (exponential scale): %.4f\n', rmse_rw1);

%% Plot the actual_value and predicted_value
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_predictions_rw, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('Real GDP');
title('Rolling Window: Random Walk Forecast vs Actual Values');
legend;
grid on;
hold off;



%%OPTIONAL：RW(using first order difference

% Initialize the arrays
predictions_diff_rw = NaN(num_predictions, 1);
restored_predictions_rw = NaN(num_predictions, 1);

for i = 1:num_predictions
    % training window
    train_data = dyt(idx1 + i - window_size : idx1 + i - 1);
    
    % RW without drift(dyt=dyt-1
    predictions_diff_rw(i) = train_data(end);     
    % restored the difference back to lnGDP
    if i == 1
%start from the initial value
        restored_predictions_rw(i) = initial_value + predictions_diff_rw(i); 
    else
        restored_predictions_rw(i) =actual_values_yt(i-1) + predictions_diff_rw(i);
    end
end

% Exponential of the predictions
exp_restored_predictions_rw = exp(restored_predictions_rw);

% %Compute RMSE
rmse_rw2 = sqrt(mean((actual_values_GDPt  - exp_restored_predictions_rw).^2));
fprintf('RMSE (exponential scale): %.4f\n', rmse_rw2);


% %Plot the actual_value and predicted_value
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_restored_predictions_rw, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('Real GDP');
title('Rolling Window: Random Walk Forecast vs Actual Values');
legend;
grid on;
hold off;

%%Model 2: AR(4)
% Initialize the arrays
predictions_diff_ar4 = NaN(num_predictions, 1);
restored_predictions_ar4 = NaN(num_predictions, 1);

for i = 1:num_predictions
    % training window
    train_data = dyt(idx1 + i - window_size : idx1 + i - 1);
    
    % AR(4) model
    ar_model = arima('Constant', 0, 'ARLags', 1:4); % 
    ar_fitted = estimate(ar_model, train_data, 'Display', 'off');
    [predictions_diff_ar4(i), ~] = forecast(ar_fitted, 1, 'Y0', train_data);
    
    % restored the difference back to lnGDP
    if i == 1
        restored_predictions_ar4(i) = initial_value + predictions_diff_ar4(i);     else
        restored_predictions_ar4(i) = actual_values_yt(i-1) + predictions_diff_ar4(i);
    end
end

% Exponential of actual values and predictions
exp_restored_predictions_ar4 = exp(restored_predictions_ar4);

% Compute RMSE
rmse_exp_ar4 = sqrt(mean((actual_values_GDPt  - exp_restored_predictions_ar4).^2));
fprintf('RMSE (exponential scale): %.4f\n', rmse_exp_ar4);

% %Plot the actual_value and predicted_value
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_restored_predictions_ar4, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('lnGDP');
title('Rolling Window: AR(4) Forecast vs Actual Values');
legend;
grid on;
hold off;
%%Model 3: VAR(4)
% Initialize the arrays
predictions_diff_var4 = NaN(num_predictions, 1);
restored_predictions_var4 = NaN(num_predictions, 1);

for i = 1:num_predictions
    % training window
     train_data = [dyt(idx1 + i - window_size:idx1 + i - 1), ...
                  pi_t(idx1 + i - window_size:idx1 + i - 1), ...
                  T_spread(idx1 + i - window_size:idx1 + i - 1)];    
    % VAR(4) model
var4_model = varm(3, 4); 
var4_fitted = estimate(var4_model, train_data);  
  
% Forecast the first variable (yt's difference)
    next_prediction = forecast(var4_fitted, 1, train_data);
    predictions_diff_var4(i, 1) = next_prediction(1, 1); 

% Restore yt from the differences
    if i == 1
        restored_predictions_var4(i) = initial_value + predictions_diff_var4(i, 1);
    else
        restored_predictions_var4(i) = actual_values_yt(i-1) + predictions_diff_var4(i, 1);
    end
end

% Exponential of actual values and predictions
exp_restored_predictions_var4 = exp(restored_predictions_var4);

% Compute RMSE
rmse_exp_var4 = sqrt(mean((actual_values_GDPt  - exp_restored_predictions_var4).^2));
fprintf('RMSE (exponential scale): %.4f\n', rmse_exp_var4);


% %Plot the actual_value and predicted_value
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_restored_predictions_var4, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('lnGDP');
title('Rolling Window: VAR(4) Forecast vs Actual Values');
legend;
grid on;
hold off;
%%Model 4: VAR(Optimal lag)
% Initialize the arrays
predictions_diff_varop = NaN(num_predictions, 1);
restored_predictions_varop = NaN(num_predictions, 1);
predictions_pit_varop = NaN(num_predictions, 1);
predictions_Tt_varop= NaN(num_predictions, 1);


% Parameters
maxLag = 8; % 
minLag = 1; % 
lag_orders = NaN(num_predictions, 1); % 

% Rolling window training set
for i = 1:num_predictions
    try
        % extract rolling window training window
        train_data = [dyt(idx1 + i - window_size:idx1 + i - 1), ...
                      pi_t(idx1 + i - window_size:idx1 + i - 1), ...
                      T_spread(idx1 + i - window_size:idx1 + i - 1)];
        T = size(train_data, 1); 
        
        % Initialize optimal AIC and Lag order
        bestAIC = Inf; 
        bestLag = NaN; 
        
        % Ergodic lag order
        for lag = minLag:maxLag
            try
                % Var(lag)model
                model = varm(3, lag);                
fittedModel = estimate(model, train_data);
                
                % calculate residuals
                residuals = infer(fittedModel, train_data); 
                SSRk = sum(residuals.^2, 'all'); % 
                
                % skip useless SSR values
                if SSRk <= 0
                    continue;
                end
                
                % calculate AIC
                AICk = log(SSRk) / T + 2 * lag / T;
                
                % updating the optimal models according to AIC
                if AICk < bestAIC
                    bestAIC = AICk;
                    bestLag = lag;
                end
            catch ME
                % If the model for the lag order cannot be estimated, skip
                fprintf('Error at iteration %d, lag %d: %s\n', i, lag, ME.message);
                continue;
            end
        end
        
        % record the optimal lag order in the current window
        lag_orders(i) = bestLag;
    catch ME
        fprintf('Error in rolling window iteration %d: %s\n', i, ME.message);
    end
end

% display optimal lag orders for all forecasting period
fprintf('Dynamic lag orders:\n');
disp(lag_orders);

for i = 1:num_predictions
    % choose the current lag
    currentLag = lag_orders(i);
    
    % extract the rolling window 
    train_data = [dyt(idx1 + i - window_size:idx1 + i - 1), ...
                  pi_t(idx1 + i - window_size:idx1 + i - 1), ...
                  T_spread(idx1 + i - window_size:idx1 + i - 1)];
    
    % VAR(3,currentLag) 
    varop_model = varm(3, currentLag); 
    varop_fitted = estimate(varop_model, train_data);
    
    % one step prediction
    next_prediction = forecast(varop_fitted, 1, train_data);
    
    % save the predictions
    predictions_diff_varop(i) = next_prediction(1, 1); % dyt
    predictions_pit_varop(i) = next_prediction(1, 2); % pi_t
    predictions_Tt_varop(i) = next_prediction(1, 3); % T_spread

   % Restore yt from the differences
    if i == 1
        restored_predictions_varop(i) = initial_value + predictions_diff_varop(i);
    else
        restored_predictions_varop(i) = actual_values_yt(i-1)+ predictions_diff_varop(i);
    end
end

% Exponential of actual values and predictions
exp_restored_predictions_varop = exp(restored_predictions_varop);

%%RMSE calculation 
rmse_varop_GDP = sqrt(mean((actual_values_GDPt- exp_restored_predictions_varop).^2));
fprintf('RMSE (GDP): %.4f\n', rmse_varop_GDP);

%% Plot for GDP,pit,T_spread
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_restored_predictions_varop, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('GDP');
title('Rolling Window: Forecast vs Actual Values (Real GDP)');
legend;
grid on;
hold off;

% Plot for π_t (Inflation)
figure;
plot(dates(idx1+1:idx2), actual_values_pit, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), predictions_pit_varop, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('π_t');
title('Rolling Window: Forecast vs Actual Values (Inflation)');
legend;
grid on;
hold off;

% Plot for T_spread (Term Spread)
figure;
plot(dates(idx1+1:idx2), actual_values_Tt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), predictions_Tt_varop, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('T_{spread}');
title('Rolling Window: Forecast vs Actual Values (Term Spread)');
legend;
grid on;
hold off;

%%Plot the residuals of three series
% Compute residuals for all three series: dyt, pi_t, T_spread
residuals_dyt=predictions_diff_varop-actual_values_yt;
residuals_pit=predictions_pit_varop-actual_values_pit; 
residuals_Tt=predictions_Tt_varop-actual_values_Tt; 

% Plot residuals for dyt, pi_t, and T_spread on the same graph
figure;
plot(dates(idx1+1:idx2), residuals_dyt, 'b-', 'DisplayName', 'dyt Residuals');
hold on;
plot(dates(idx1+1:idx2), residuals_pit, 'r-', 'DisplayName', 'pi_t Residuals');
plot(dates(idx1+1:idx2), residuals_Tt, 'g-', 'DisplayName', 'T_spread Residuals');
xlabel('Date');
ylabel('Residuals');
title('Residuals of dyt, pi_t, and T_spread');
legend;
grid on;
hold off;

%% Plot ACF and PACF for each variable's residuals
residuals_3 = table(residuals_dyt,residuals_pit, residuals_Tt, ...
               'VariableNames', {'res_dyt', 'res_pit', 'res_T_spread'});
variable_names = {'dyt Residuals', 'pi_t Residuals', 'T_spread Residuals'};
% Loop through each residual column and plot ACF and PACF
for i = 1:3
    figure;

    % ACF
    subplot(2, 1, 1);
    autocorr(residuals_3(:, i), 'NumLags', 20);
    title(['ACF of Residuals: ', variable_names{i}]);

    % PACF
    subplot(2, 1, 2);
    parcorr(residuals_3(:, i), 'NumLags', 20);
    title(['PACF of Residuals: ', variable_names{i}]);
end



%%Model 5: AR(4)-X
X=datafactor(1:idx1+1, :);
Time=X(:,1);
X(:,6)=[];
X(:,1)=[];
PCdata=table2array(X);
Xs = zscore(PCdata); % standardize
[coeff,F,~,~,explained] = pca(Xs);

bar(1:10,explained(1:10))

time_values = Time.Time;
plot(time_values, movmean(F(:, 1), 4)); 
title('(4 quarters avg.) First PC ($\hat{F}_{1})$', 'Interpreter', 'latex');
recessionplot;

K = size(Xs,2);
for kk=1:K
 res = fitlm(Xs(:,kk),F(:,1));
 R2(kk,1) = res.Rsquared.Ordinary;
 res = fitlm(Xs(:,kk),F(:,2));
 R2(kk,2) = res.Rsquared.Ordinary;
 res = fitlm(Xs(:,kk),F(:,3));
 R2(kk,3) = res.Rsquared.Ordinary;
end

[~,id] = sort(R2(:,1),1,"descend");
bar(1:10, R2(id(1:10),1)'); xticks(1:10); 
xticklabels(X.Properties.VariableNames(id(1:10))); xtickangle(45)
[~,id] = sort(R2(:,2),1,"descend");
bar(1:10, R2(id(1:10),2)'); xticks(1:10); 
xticklabels(X.Properties.VariableNames(id(1:10))); xtickangle(45)
[~,id] = sort(R2(:,3),1,"descend");
bar(1:10, R2(id(1:10),3)'); xticks(1:10); 
xticklabels(X.Properties.VariableNames(id(1:10))); xtickangle(45)

TimeArray = Time{1:end, 1};
figure;
plot(TimeArray, movmean(zscore(dyt(1:idx1+1, 1)), 4), '-b', ... 
     TimeArray, movmean(zscore(F(:, 1)), 4), '-r');           
legend('dyt', 'Factor 1');
xlabel('Time');
ylabel('Standardized Values');
title('Smoothed Standardized dyt vs. Factor 1');
grid on;

XX=datafactor(1:idx2, :);
XX(:,6)=[];
XX(:,1)=[];
PC_diff = diff(XX);
PCdata_all=table2array(PC_diff);

% Initialize the arrays
window_size = 99;
predictions_diff_arpc = NaN(num_predictions, 1);
restored_predictions_arpc = NaN(num_predictions, 1);

for i = 1:num_predictions
    % Define the training window for both dyt and PC data
    train_data_dyt = dyt(idx1 + i - window_size:idx1 + i - 1); 
    train_data_F = datafactor(idx1 + i - window_size:idx1 + i - 1, :);  % Use the full table for PCA
    
    % Remove the 1st and 6th columns from the table for PCA calculation
    train_data_F = train_data_F(:, 2:end); % This will remove the first column (assuming the second column is used for PCA)
    train_data_F(:, 5) = [];  % Removing the 6th column (index 5, as indexing starts from 1)
    
    % Ensure standardized PCA input
Xs_all = zscore(table2array(train_data_F)); % Convert to array and standardize


F1 = F(:,1);

% Lag matrix consistency check
lags_dyt = lagmatrix(train_data_dyt, 1:4);
lstart_idx = idx1 + i - window_size;
end_idx = idx1 + i - 1;

% Safety check
if lstart_idx < 1 || end_idx > size(F1, 1)
    warning('Skipping iteration %d due to index bounds.', i);
    continue;
end

lagged_F1_window = lagmatrix(F1(lstart_idx:end_idx, 1), 1);
 % Ensure use of the first PC

% Filtering valid rows
valid_idx = ~any(isnan([lags_dyt, lagged_F1]), 2);

% Model fitting
if sum(valid_idx) >= 5 % Ensure enough data points for regression
    X_train = [ones(sum(valid_idx), 1), lags_dyt(valid_idx, :), lagged_F1(valid_idx)];
    y_train = train_data_dyt(valid_idx);
    beta = X_train \ y_train;
else
    warning('Not enough data for regression in this window.');
    beta = NaN(size(X_train, 2), 1); % Handle sparse data windows
end

    
    % One-step forecast
    next_lags_dyt = train_data_dyt(end-3:end)'; 
    next_F1 = F1(end); 
    X_next = [1, next_lags_dyt, next_F1]; 
    predictions_diff_arpc(i) = X_next * beta; 
    
    % Restore yt from the differences
    if i == 1
        restored_predictions_arpc(i) = initial_value + predictions_diff_arpc(i); 
    else
        restored_predictions_arpc(i) = actual_values_yt(i-1) + predictions_diff_arpc(i); 
    end
end

% Exponential of actual values and predictions
exp_restored_predictions_arpc = exp(restored_predictions_arpc);

%%RMSE calculation 
rmse_arpc_GDP = sqrt(mean((actual_values_GDPt- exp_restored_predictions_arpc).^2));
fprintf('RMSE (GDP): %.4f\n', rmse_arpc_GDP);

%% Plot the actual_value and predicted_value
figure;
plot(dates(idx1+1:idx2), actual_values_GDPt, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(dates(idx1+1:idx2), exp_restored_predictions_arpc, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Date');
ylabel('Real GDP');
title('Rolling Window: AR(4)-ex Forecast vs Actual Values');
legend;
grid on;
hold off;


% Precomputed RMSE values
rmseModel1 = 119.43; 
rmseModel2 = 83.77; 
rmseModel3 = 93.37; 
rmseModel4 = 84.77
rmseModel5 = 85.13; 

% Model names
modelNames = ["Random Walk"; "AR(4)"; "VAR(4)"; "AR(4)-X"; "VAR Optimal Lag"];

% Combine RMSE values into an array
rmseValues = [rmseModel1; rmseModel2; rmseModel3; rmseModel4; rmseModel5];

% Create the table
rmseTable = table(modelNames, rmseValues, 'VariableNames', {'Model', 'RMSE'});

% Display the table
disp(rmseTable);