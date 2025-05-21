%% register across timepoints and compute CoV variance
clear;clc
loadloc = 'F:\00_CoV';
disease = 'CONTROL';
subject =  'CV003';%'CV001';
datasize = [300,1536,300];

proj_layer = 'NFL_ONL';

numPoints = datasize(2);
numBscans = datasize(3);
numAscans = datasize(1);

% for en face OCTA binarization
DNN_model = 'F:\CoV_Processing\CoV_GUI'; % Neural network location

save_folder = fullfile(loadloc, disease, subject, 'Results3D');
%%
tps = {'OD_MAC_12190100PM','OD_MAC_12200100PM','OD_MAC_01090100PM','OD_MAC_01100100PM'};
tp_template = 'OD_MAC_12190100PM';
register_across_tps(loadloc, disease, subject, datasize, tps, tp_template, proj_layer);
%% compute variation
save_comp_reg = fullfile(save_folder,'TemporalAnlysis',proj_layer, 'Registered');
tps_200 = {'OD_MAC_12190100PM','OD_MAC_12200100PM','OD_MAC_01090100PM','OD_MAC_01100100PM'};

% savefns_temp = fullfile(save_folder, tp_template);%,'Extraction');
% save_mat_temp = fullfile(savefns_temp,'mat');
% save_nifti_temp = fullfile(savefns_temp,'nifti');
% save_mask_fn_temp = fullfile(save_nifti_temp,'avg_mask_all_corr.nii');
% avg_mask_all_temp = double(niftiread(save_mask_fn_temp));
% avg_mask_all_template = avg_mask_all_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);

cropped_cov = 15;
projCoV_fromZeiss = [];
projCov_fromVolume = [];
projCoV_mean= [];
projCoV_median = [];

vessel_mask_all_temp = importdata(fullfile(save_comp_reg,'averaged_vessel_mask.mat'));
vessel_mask_all = vessel_mask_all_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);

imshow(vessel_mask_all,[])


for t = 1:length(tps_200)
    projCoV_fromZeiss_temp = importdata(fullfile(save_comp_reg,strcat(tps_200{t},'_registered_projCoV_fromZeiss.mat')));
    projCoV_fromZeiss(:,:,t) = projCoV_fromZeiss_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);

    projCoV_fromVolume_temp = importdata(fullfile(save_comp_reg,strcat(tps_200{t},'_registered_projCoV_fromVolume.mat')));
    projCov_fromVolume(:,:,t) =  projCoV_fromVolume_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);

    projCoV_mean_temp = importdata(fullfile(save_comp_reg,strcat(tps_200{t},'_registered_projCoV_mean.mat')));
    projCoV_mean(:,:,t) = projCoV_mean_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);

    projCoV_median_temp = importdata(fullfile(save_comp_reg,strcat(tps_200{t},'_registered_projCoV_median.mat')));
    projCoV_median(:,:,t) = projCoV_median_temp(cropped_cov+1:end-cropped_cov,cropped_cov+1:end-cropped_cov);
end

%% Statistical way to find the turning point
data_temp = projCoV_median(:,:,4).*vessel_mask_all;
% Step 1: Prepare the Data
flattened_data = nonzeros(data_temp(:));

%%

% Compute the histogram
[counts, bin_edges] = histcounts(flattened_data, 'Normalization', 'pdf');
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;

% Step 2: Define the Lognormal Mixture Model
lognormal_pdf = @(x, mu, sigma) (1 ./ (x * sigma * sqrt(2 * pi))) .* exp(-((log(x) - mu).^2) ./ (2 * sigma^2));
lognormal_mixture = @(params, x) ...
    params(1) * lognormal_pdf(x, params(2), params(3)) + ...
    params(4) * lognormal_pdf(x, params(5), params(6));

% Step 3: Initial Guess for Parameters
% [weight1, mu1, sigma1, weight2, mu2, sigma2]
initial_params = [0.5, log(mean(flattened_data)), 0.5, 0.5, log(mean(flattened_data) * 1.5), 0.5];

% Step 4: Fit the Lognormal Mixture Model
% Use lsqcurvefit to optimize the parameters
options = optimoptions('lsqcurvefit', 'Display', 'off');
fitted_params = lsqcurvefit(@(params, x) lognormal_mixture(params, x), ...
                            initial_params, bin_centers, counts, [], [], options);

% Step 5: Extract the Fitted Parameters
weight1 = fitted_params(1);
mu1 = fitted_params(2);
sigma1 = fitted_params(3);
weight2 = fitted_params(4);
mu2 = fitted_params(5);
sigma2 = fitted_params(6);

% Step 6: Compute the Individual and Combined PDFs
pdf1 = weight1 * lognormal_pdf(bin_centers, mu1, sigma1);
pdf2 = weight2 * lognormal_pdf(bin_centers, mu2, sigma2);
pdf_combined = pdf1 + pdf2;

% Step 7: Plot the Results
figure;
% Plot the histogram
histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
% Plot the individual lognormal components
plot(bin_centers, pdf1, 'r--', 'LineWidth', 2);
plot(bin_centers, pdf2, 'b--', 'LineWidth', 2);
% Plot the combined PDF
plot(bin_centers, pdf_combined, 'g-', 'LineWidth', 2);

% Add labels, legend, and title
legend('Histogram', 'Lognormal Component 1', 'Lognormal Component 2', 'Combined Lognormal Mixture');
title('Histogram Fit with Two Lognormal Functions');
xlabel('x');
ylabel('Density');
hold off;

% Step 8: Display Fitted Parameters
disp('Fitted Parameters:');
disp(table(weight1, mu1, sigma1, weight2, mu2, sigma2, ...
    'VariableNames', {'Weight1', 'Mu1', 'Sigma1', 'Weight2', 'Mu2', 'Sigma2'}));

%% Fit GMM with 2 clusters (all points)
% Step 2: Fit the Gaussian Mixture Model (GMM)
rng(123); % Set random seed for reproducibility
% gm = fitgmdist(flattened_data, 3); % Fit GMM with 2 components
options = statset('MaxIter', 1000); % Increase maximum iterations
gm = fitgmdist(flattened_data, 2, 'Replicates', 10, 'Options', options); % Use multiple initializations

% Extract GMM parameters
mu1 = gm.mu(1); % Mean of the first Gaussian
mu2 = gm.mu(2); % Mean of the second Gaussian
sigma1 = sqrt(gm.Sigma(1)); % Standard deviation of the first Gaussian
sigma2 = sqrt(gm.Sigma(2)); % Standard deviation of the second Gaussian
weight1 = gm.ComponentProportion(1); % Weight of the first Gaussian
weight2 = gm.ComponentProportion(2); % Weight of the second Gaussian

% Define the Gaussian functions
f1 = @(x) weight1 * (1 / (sigma1 * sqrt(2 * pi))) * exp(-((x - mu1).^2) / (2 * sigma1^2));
f2 = @(x) weight2 * (1 / (sigma2 * sqrt(2 * pi))) * exp(-((x - mu2).^2) / (2 * sigma2^2));

% Step 3: Define the Difference Function
f_diff = @(x) f1(x) - f2(x);

% Step 4: Identify All Intersections
% Define the search range based on data
x_vals = linspace(min(flattened_data), max(flattened_data), 1000); % Fine grid for scanning
f_vals = arrayfun(f_diff, x_vals); % Evaluate the difference function

% Detect sign changes
sign_change_indices = find(diff(sign(f_vals)) ~= 0);

% Initialize storage for intersection points
intersection_points = [];

% Find all roots using fzero
for i = 1:length(sign_change_indices)
    % Define the interval around the sign change
    x_start = x_vals(sign_change_indices(i));
    x_end = x_vals(sign_change_indices(i) + 1);

    % Find the intersection point in this interval
    try
        intersection_point = fzero(f_diff, [x_start, x_end]);
        intersection_points = [intersection_points; intersection_point];
    catch
        warning(['No intersection found in interval: [', num2str(x_start), ', ', num2str(x_end), ']']);
    end
end

% Evaluate the Gaussians at the intersection points
values_f1 = f1(intersection_points);
values_f2 = f2(intersection_points);

% Step 5: Visualize the Results
% Compute Gaussian values for plotting
pdf1 = f1(x_vals);
pdf2 = f2(x_vals);

% Compute the scaling factor for the Gaussian with the largest peak
data_peak = max(pdf_values); % Peak of the KDE or data
gmm_peak = max(pdf1); % Peak of the largest Gaussian
scaling_factor = data_peak / gmm_peak;

% Rescale the Gaussian component
pdf1_scaled = pdf1 * scaling_factor;

% Plot the GMM components and the intersection points
figure;
histogram(flattened_data, 'Normalization', 'pdf','NumBins',100); hold on; % , 'DisplayStyle', 'stairs', 'EdgeColor', 'k'
plot(x_vals, pdf1_scaled, 'r', 'LineWidth', 2);
plot(x_vals, pdf2, 'b', 'LineWidth', 2);
plot(intersection_points, values_f1, 'go', 'MarkerSize', 10, 'LineWidth', 2);
legend('Data Histogram', 'Gaussian 1', 'Gaussian 2', 'Intersection Points');
title('GMM Fit and Intersection Points');
xlabel('Data Values');
ylabel('Density');
hold off;

% Step 6: Display Results
disp('All Intersection Points:');
disp(table(intersection_points, values_f1, values_f2, ...
    'VariableNames', {'Intersection_X', 'Gaussian1_Value', 'Gaussian2_Value'}));
%% Updated combined GMM and KDE (using intersection points as the turning points)
% Step 2: Define a Common x-Value Grid
x_vals = linspace(min(flattened_data), max(flattened_data), length(flattened_data)); % 1000 points over the data range

% Step 3: Compute KDE on the Common Grid
[pdf_values, ~] = ksdensity(flattened_data, x_vals); % KDE using the common grid

% Step 4: Fit the GMM
rng(123); % Set random seed for reproducibility
gm = fitgmdist(flattened_data, 2); % Fit a Gaussian Mixture Model with 2 components

% Step 5: Compute GMM PDF on the Common Grid
pdf1 = gm.ComponentProportion(1) * normpdf(x_vals, gm.mu(1), sqrt(gm.Sigma(1)));
pdf2 = gm.ComponentProportion(2) * normpdf(x_vals, gm.mu(2), sqrt(gm.Sigma(2)));
pdf_combined = pdf1 + pdf2;

% Step 6: Find the Intersection Point
% Step 1: Define the Difference Function
f_diff = @(x) interp1(x_vals, pdf_values, x, 'linear', 0) - interp1(x_vals, pdf_combined, x, 'linear', 0);

% Step 2: Identify Regions with Sign Changes
% Evaluate the difference function on the grid
f_vals = arrayfun(f_diff, x_vals);

% Find indices where the function changes sign
sign_change_indices = find(diff(sign(f_vals)) ~= 0);

% Initialize a list to store all intersection points
intersection_points = [];

% Step 3: Refine Each Intersection with fzero
for i = 1:length(sign_change_indices)
    % Define the interval around the sign change
    x_start = x_vals(sign_change_indices(i));
    x_end = x_vals(sign_change_indices(i) + 1);

    % Use fzero to find the root in this interval
    intersection_point = fzero(f_diff, [x_start, x_end]);
    intersection_points = [intersection_points; intersection_point]; % Store the result
end

% Step 4: Evaluate KDE and GMM Values at the Intersections
kde_values = interp1(x_vals, pdf_values, intersection_points, 'linear', 0);
gmm_values = interp1(x_vals, pdf_combined, intersection_points, 'linear', 0);

% Display Results
disp('Intersection Points and Corresponding Values:');
disp(table(intersection_points, kde_values, gmm_values, ...
    'VariableNames', {'Intersection_X', 'KDE_Value', 'GMM_Value'}));

% Step 5: Plot the Results
figure;
% Plot the histogram, KDE, and GMM PDF
histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); % KDE
plot(x_vals, pdf_combined, 'r--', 'LineWidth', 2); % GMM combined PDF

% Highlight all intersection points
plot(intersection_points, kde_values, 'go', 'MarkerSize', 10, 'LineWidth', 2);

% Add labels, legend, and title
legend('Histogram', 'KDE Estimate', 'Combined GMM PDF', 'Intersection Points');
title('All Intersection Points of KDE and GMM PDF');
xlabel('x');
ylabel('Density');
hold off;

%% GMM and KDE combined but define the turning points as the location where slop are the same (close enough) to each other
% Define a common x-value grid
x_vals = linspace(min(flattened_data), max(flattened_data), length(flattened_data)); % Match grid size to data

% Step 2: Compute KDE on the Common Grid
[pdf_values, ~] = ksdensity(flattened_data, x_vals); % KDE using the common grid

% Step 3: Fit the Gaussian Mixture Model (GMM)
rng(123); % Set random seed for reproducibility
gm = fitgmdist(flattened_data, 2); % Fit GMM with 2 components

% Step 4: Compute GMM PDF on the Common Grid
pdf1 = gm.ComponentProportion(1) * normpdf(x_vals, gm.mu(1), sqrt(gm.Sigma(1)));
pdf2 = gm.ComponentProportion(2) * normpdf(x_vals, gm.mu(2), sqrt(gm.Sigma(2)));
pdf_combined = pdf1 + pdf2;

% Step 5: Compute the Derivatives (Slopes)
% Numerical differentiation for both KDE and GMM
dx = mean(diff(x_vals)); % Compute grid spacing
kde_slope = diff(pdf_values) / dx; % Slope of the KDE
gmm_slope = diff(pdf_combined) / dx; % Slope of the GMM
x_vals_slope = x_vals(1:end-1); % Corresponding x-values for slopes

% Step 6: Identify Points Within the Slope Difference Threshold
% Define the slope difference threshold
threshold = 0.5; 

% Compute the absolute slope difference
slope_diff = abs(kde_slope - gmm_slope);

% Find indices where the slope difference is within the threshold
valid_indices = find(slope_diff <= threshold);

% Record the corresponding x-values and slope difference values
recorded_points_x = x_vals_slope(valid_indices);
recorded_slope_diff = slope_diff(valid_indices);

% Step 7: Evaluate the KDE and GMM at the Recorded Points
kde_values_at_points = interp1(x_vals, pdf_values, recorded_points_x, 'linear', 0);
gmm_values_at_points = interp1(x_vals, pdf_combined, recorded_points_x, 'linear', 0);

% Display Results
disp('Points Where Slope Difference is Within the Threshold:');
disp(table(recorded_points_x, kde_values_at_points, gmm_values_at_points, recorded_slope_diff, ...
    'VariableNames', {'X', 'KDE_Value', 'GMM_Value', 'Slope_Difference'}));

% Step 8: Plot the Histogram and Density Functions
figure;
histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); % KDE
plot(x_vals, pdf_combined, 'r--', 'LineWidth', 2); % GMM combined PDF

% Highlight the recorded points
plot(recorded_points_x, kde_values_at_points, 'go', 'MarkerSize', 8, 'LineWidth', 2);

% Add labels, legend, and title
legend('Histogram', 'KDE Estimate', 'Combined GMM PDF', 'Points Within Threshold');
title('Points Where KDE and GMM Slopes Are Similar');
xlabel('x');
ylabel('Density');
hold off;

% Step 9: Plot the Slope Difference and Highlight Points
figure;
plot(x_vals_slope, slope_diff, 'k', 'LineWidth', 2); hold on;
plot(recorded_points_x, recorded_slope_diff, 'go', 'MarkerSize', 8, 'LineWidth', 2);

% Add labels, legend, and title
legend('Slope Difference', 'Points Within Threshold');
title('Slope Difference Between KDE and GMM');
xlabel('x');
ylabel('Absolute Slope Difference');
hold off;

%% Method 3: Combined model but selecting the points where KDE no longer follows GMM
% Define a common x-value grid
x_vals = linspace(min(flattened_data), max(flattened_data), length(flattened_data)); % Match grid size to data

% Step 2: Compute KDE on the Common Grid
[pdf_values, ~] = ksdensity(flattened_data, x_vals); % KDE using the common grid

% Step 3: Fit the Gaussian Mixture Model (GMM)
rng(123); % Set random seed for reproducibility
gm = fitgmdist(flattened_data, 2); % Fit GMM with 2 components

% Step 4: Compute GMM PDF on the Common Grid
pdf1 = gm.ComponentProportion(1) * normpdf(x_vals, gm.mu(1), sqrt(gm.Sigma(1)));
pdf2 = gm.ComponentProportion(2) * normpdf(x_vals, gm.mu(2), sqrt(gm.Sigma(2)));
pdf_combined = pdf1 + pdf2;

% Step 5: Compute the Derivatives (Slopes)
% Numerical differentiation for both KDE and GMM
dx = mean(diff(x_vals)); % Compute grid spacing
kde_slope = diff(pdf_values) / dx; % Slope of the KDE
gmm_slope = diff(pdf_combined) / dx; % Slope of the GMM
x_vals_slope = x_vals(1:end-1); % Corresponding x-values for slopes

% Step 6: Identify All Divergence Points
% Define the slope difference threshold
threshold = 0.25; % Replace with your desired threshold

% Compute the absolute slope difference
slope_diff = abs(kde_slope - gmm_slope);

% Find all points where the slope difference exceeds the threshold
divergence_indices_all = find(slope_diff > threshold);

% Record all divergence points
divergence_points_x = x_vals_slope(divergence_indices);
kde_values_at_divergence = pdf_values(divergence_indices);
gmm_values_at_divergence = pdf_combined(divergence_indices);
slope_differences_at_divergence = slope_diff(divergence_indices);

% Display Results
disp('All Divergence Points:');
disp(table(divergence_points_x, kde_values_at_divergence, gmm_values_at_divergence, slope_differences_at_divergence, ...
    'VariableNames', {'X', 'KDE_Value', 'GMM_Value', 'Slope_Difference'}));

% Step 7: Plot the Histogram and Density Functions
figure;
histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); % KDE
plot(x_vals, pdf_combined, 'r--', 'LineWidth', 2); % GMM combined PDF

% Highlight all divergence points
plot(divergence_points_x, kde_values_at_divergence, 'go', 'MarkerSize', 8, 'LineWidth', 2);

% Add labels, legend, and title
legend('Histogram', 'KDE Estimate', 'Combined GMM PDF', 'Divergence Points');
title('Divergence Points Where KDE No Longer Follows GMM');
xlabel('x');
ylabel('Density');
hold off;

% Step 8: Plot the Slope Difference
figure;
plot(x_vals_slope, slope_diff, 'k', 'LineWidth', 2); hold on;

% Highlight all divergence points on the slope difference plot
plot(divergence_points_x, slope_differences_at_divergence, 'go', 'MarkerSize', 8, 'LineWidth', 2);

% Add labels, legend, and title
legend('Slope Difference', 'Divergence Points');
title('Slope Difference Between KDE and GMM');
xlabel('x');
ylabel('Absolute Slope Difference');
hold off;

%% plot and threshold
data_temp = projCoV_median(:,:,1).*vessel_mask_all;
% Step 1: Prepare the Data
flattened_data = nonzeros(data_temp(:));

%%% Median CoV from Bscans
cm = getcolormap_sup();
q = strcat("what is the threshold value: ");
% close all
% fmedian_hist = figure;
figure('Visible','off');
histogram(data_temp,'NumBins',150,'Normalization','pdf');
title('CoV values from B-scan basis CoV computation','FontName','Times New Roman','FontSize',12);%'
xlabel('Pixel Values [a.u.]','FontName','Times New Roman')
ylabel('Normalized # of Pixels','FontName','Times New Roman')
xlim([0,1])
% ylim([0,1200])
fmedian_hist = gca;
xt = get(fmedian_hist, 'YTick');
set(fmedian_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))

ptitle = 'CoV values from B-scan basis CoV computation';
threshold_Median = inputDialogWithPlot(data_temp, ptitle, q);% input("What is the threshold value: ");

data_temp(data_temp > threshold_Median) = threshold_Median; % covBscan_MED_mask_norm

fmedian = figure('Visible','off');
% savefn_img = fullfile(app.save_comp,strcat(app.tp,'_CoV_proj_fromVolume_median_threshold_',strrep(num2str(threshold_Median),'.','-'),'.png'));
imshow(ind2rgb(round(mat2gray(data_temp).*255),cm),cm);%title("CoV of Median OCTA Bscan");cbar_median = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0',num2str(threshold_Median)});
% exportgraphics(fmedian,savefn_img)
% close all

%%% differenciate vessels at different layers
data_temp_sup = data_temp;

% q = strcat(app.tp," - what is the threshold value to separate the deep vessels: ");
deep_MED_threshold = inputDialogWithPlot(data_temp, ptitle, q);% inputdlg("What is the threshold value for deep vessels: ");
data_temp_sup(data_temp_sup > deep_MED_threshold) = 0; % (deep_MED_threshold/threshold_Median) if normalized


fmedian_sup = figure('Visible','on');
imshow(ind2rgb(round((data_temp_sup).*255),cm),cm); % mat2gray
% title(["Median CoV of Superficial Vessel",...
%     "from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_MED_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_MED_threshold/threshold_Mean,2)),'1'});
ax = gca;
ax.TitleFontSizeMultiplier = 0.8;
% savefn_img = fullfile(app.save_comp,strcat(app.tp,'_supVessel_CoV_proj_fromVolume_median_threshold_',strrep(num2str(deep_MED_threshold),'.','-'),'.png'));
% exportgraphics(fmedian_sup,savefn_img)
% close all

data_temp_deep = data_temp;
data_temp_deep(data_temp_deep <= deep_MED_threshold) = 0; % (deep_MED_threshold/threshold_Median) if mat2gray

fmedian_deep = figure('Visible','on');
imshow(ind2rgb(round((data_temp_deep).*255),cm),cm); % mat2gray
% % title(["Median CoV of Deep Vessel",...
% %     "from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_MED_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_MED_threshold/threshold_Mean,2)),'1'});
ax = gca;
ax.TitleFontSizeMultiplier = 0.8;
% % savefn_img = fullfile(app.save_comp,strcat(app.tp,'_deepVessel_CoV_proj_fromVolume_median_threshold_',strrep(num2str(deep_MED_threshold),'.','-'),'.png'));
% exportgraphics(fmedian_deep,savefn_img)
% close all

% Define the difference function
f_diff = @(x) interp1(x_vals, pdf_values, x, 'linear', 0) - interp1(x_vals, pdf_combined, x, 'linear', 0);

% Use fzero to find the intersection point
intersection_point = fzero(f_diff, [min(x_vals), max(x_vals)]);

% Evaluate KDE and GMM PDF at the intersection
kde_value = interp1(x_vals, pdf_values, intersection_point, 'linear', 0);
pdf_value = interp1(x_vals, pdf_combined, intersection_point, 'linear', 0);

% Step 7: Plot Results
figure;
% Plot the histogram
histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
% Plot KDE
plot(x_vals, pdf_values, 'b', 'LineWidth', 2);
% Plot GMM combined PDF
plot(x_vals, pdf_combined, 'r--', 'LineWidth', 2);
% Highlight the intersection point
plot(intersection_point, kde_value, 'go', 'MarkerSize', 10, 'LineWidth', 2);

% Add labels, legend, and title
legend('Histogram', 'KDE Estimate', 'Combined GMM PDF', 'Intersection Point');
title('Histogram, KDE, and GMM PDF with Intersection');
xlabel('x');
ylabel('Density');
hold off;

% Step 8: Display Results
disp(['Intersection Point: x = ', num2str(intersection_point)]);
disp(['KDE Value at Intersection: ', num2str(kde_value)]);
disp(['PDF Combined Value at Intersection: ', num2str(pdf_value)]);


%% one point
% % Step 2: Fit the Gaussian Mixture Model (GMM)
% rng(123); % Set random seed for reproducibility
% gm = fitgmdist(flattened_data, 2);
%
% % Extract GMM parameters
% mu1 = gm.mu(1); % Mean of the first Gaussian
% mu2 = gm.mu(2); % Mean of the second Gaussian
% sigma1 = sqrt(gm.Sigma(1)); % Standard deviation of the first Gaussian
% sigma2 = sqrt(gm.Sigma(2)); % Standard deviation of the second Gaussian
% weight1 = gm.ComponentProportion(1); % Weight of the first Gaussian
% weight2 = gm.ComponentProportion(2); % Weight of the second Gaussian
%
% % Define the Gaussian functions
% f1 = @(x) weight1 * (1 / (sigma1 * sqrt(2 * pi))) * exp(-((x - mu1).^2) / (2 * sigma1^2));
% f2 = @(x) weight2 * (1 / (sigma2 * sqrt(2 * pi))) * exp(-((x - mu2).^2) / (2 * sigma2^2));
%
% % Step 3: Define the Difference Function
% f_diff = @(x) f1(x) - f2(x);
%
% % Step 4: Check Function Values at Endpoints
% x_start = min(flattened_data); % Start of the range
% x_end = max(flattened_data);   % End of the range
%
% % Evaluate the difference function at the endpoints
% f_start = f_diff(x_start);
% f_end = f_diff(x_end);
%
% % Ensure there is a sign change
% if sign(f_start) == sign(f_end)
%     disp('No sign change detected in the entire range. Searching for intervals with sign changes...');
%
%     % Step 5: Refine the Interval Using Grid Search
%     % Generate a fine grid of x-values
%     x_vals = linspace(x_start, x_end, length(flattened_data));
%     f_vals = arrayfun(f_diff, x_vals);
%
%     % Find intervals with sign changes
%     sign_change_indices = find(diff(sign(f_vals)) ~= 0);
%
%     if isempty(sign_change_indices)
%         error('No intersection point detected: f_diff does not cross zero in the specified range.');
%     end
%
%     % Use the first interval with a sign change
%     x_start = x_vals(sign_change_indices(1));
%     x_end = x_vals(sign_change_indices(1) + 1);
% end
%
% % Step 6: Find the Intersection Point
% % Use fzero to find the root in the refined interval
% intersection_point = fzero(f_diff, [x_start, x_end]);
%
% % Evaluate the Gaussians at the intersection point
% value_f1 = f1(intersection_point);
% value_f2 = f2(intersection_point);
%
% % Step 7: Visualize the Results
% % Generate x-values for plotting
% x_vals = linspace(min(flattened_data), max(flattened_data), 1000);
%
% % Compute Gaussian values for plotting
% pdf1 = f1(x_vals);
% pdf2 = f2(x_vals);
%
% % Plot the GMM components and the intersection point
% figure;
% histogram(flattened_data, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k'); hold on;
% plot(x_vals, pdf1, 'r', 'LineWidth', 2);
% plot(x_vals, pdf2, 'b', 'LineWidth', 2);
% plot(intersection_point, value_f1, 'go', 'MarkerSize', 10, 'LineWidth', 2);
% legend('Flattened Data Histogram', 'Gaussian 1', 'Gaussian 2', 'Turning Point');
% title('GMM Fit and Turning Point');
% xlabel('Data Values');
% ylabel('Density');
% hold off;
%
% % Step 8: Display Results
% disp(['Intersection Point (Turning Point): x = ', num2str(intersection_point)]);
% disp(['Value of Gaussian 1 at Turning Point: ', num2str(value_f1)]);
% disp(['Value of Gaussian 2 at Turning Point: ', num2str(value_f2)]);
%

%% Old combined
% rng(123); % Set random seed for reproducibility
% gm = fitgmdist(flattened_data, 2); % Fit a Gaussian Mixture Model with 2 components
%
% % Extract parameters of the GMM
% mu = gm.mu; % Means of the clusters
% sigma = sqrt(squeeze(gm.Sigma)); % Standard deviations of the clusters
% weights = gm.ComponentProportion; % Initial weights of the clusters
%
% % Compute the current maximum value of the second cluster
% mu2 = mu(2); % Mean of the second cluster
% sigma2 = sigma(2); % Standard deviation of the second cluster
% current_max2 = weights(2) / (sqrt(2 * pi) * sigma2);
%
% % Define the desired maximum value for the second cluster
% desired_max2 = max(flattened_data); % Replace with your desired value
%
% % Scale the weight of the second cluster to match the desired maximum value
% scaling_factor = desired_max2 / current_max2;
% weights(2) = weights(2) * scaling_factor;
%
% % Ensure the weights still sum to 1
% weights = weights / sum(weights);
%
% % Create a new gmdistribution object with the adjusted weights
% new_gm = gmdistribution(mu, reshape(sigma.^2, [1 1 2]), weights);
%
% % Evaluate the PDFs for each component and the combined PDF
% x_vals = linspace(min(flattened_data), max(flattened_data), 1000);
% pdf1 = new_gm.ComponentProportion(1) * normpdf(x_vals, new_gm.mu(1), sqrt(new_gm.Sigma(1)));
% pdf2 = new_gm.ComponentProportion(2) * normpdf(x_vals, new_gm.mu(2), sqrt(new_gm.Sigma(2)));
% % pdf_combined = pdf1 + pdf2;
%
% % Plot the results
% figure;
% histogram(flattened_data,'Normalization','pdf'); hold on;
% plot(x_vals, pdf_combined, 'k--', 'LineWidth', 2);
% legend('Combined PDF', 'Desired Peak');
% title('GMM Components with Adjusted Peak');
% xlabel('x');
% ylabel('Density');
% hold off;
%
% % Display confirmation of successful scaling
% disp(['The maximum value of the second cluster was adjusted to: ', num2str(desired_max2)]);
% %% KDE estimation
% % Perform Kernel Density Estimation (KDE) for the data
% [pdf_values, ~] = ksdensity(flattened_data,'NumPoints', 1000);
%
% % Ensure at least two peaks exist
% if numel(peak_locs) < 2
%     error('KDE found less than two peaks, indicating insufficient clustering.');
% end
%
% % Plot the results
% figure;
% plot(x_vals, pdf_values, 'b', 'LineWidth', 2);
% title('KDE Estimation');
% xlabel('Data Values');
% ylabel('Density');
% hold off;
% %% compute the intersection
% % Define the difference function between KDE and combined PDF
% f_diff = @(x) interp1(x_vals, pdf_values, x, 'linear', 0) - interp1(x_vals, pdf_combined, x, 'linear', 0);
%
% % Define an initial search range for fzero (adjust as needed)
% x_start = min(flattened_data);
% x_end = max(flattened_data);
%
% % Use fzero to find the root (intersection point)
% intersection_point = fzero(f_diff, [x_start, x_end]);
%
% % Evaluate the KDE and GMM PDF at the intersection
% kde_value = interp1(x_vals, pdf_values, intersection_point, 'linear', 0);
% pdf_value = interp1(x_vals, pdf_combined, intersection_point, 'linear', 0);
%
% % Plot KDE and combined PDF
% figure;
% plot(x_vals, pdf_values, 'b', 'LineWidth', 2); hold on;
% plot(x_vals, pdf_combined, 'r--', 'LineWidth', 2);
%
% % Highlight the intersection point
% plot(intersection_point, kde_value, 'go', 'MarkerSize', 10, 'LineWidth', 2);
%
% histogram(flattened_data,'Normalization','pdf');
%
% % Add labels and legend
% legend('KDE Estimate', 'Combined GMM PDF', 'Intersection Point');
% title('Intersection of KDE and Combined GMM PDF');
% xlabel('x');
% ylabel('Density');
% hold off;
%
% % Display the intersection point
% disp(['Intersection Point: x = ', num2str(intersection_point)]);
% disp(['KDE Value at Intersection: ', num2str(kde_value)]);
% disp(['PDF Combined Value at Intersection: ', num2str(pdf_value)]);


%% Gaussian (two clusters)
% options = statset('MaxIter', 1000); % Increase maximum iterations
gm = fitgmdist(flattened_data, 2);%, 'Options', options,'Replicates', 100);

mu1 = gm.mu(1);
mu2 = gm.mu(2);
sigma1 = sqrt(gm.Sigma(1));
sigma2 = sqrt(gm.Sigma(2));
weight1 = gm.ComponentProportion(1);
weight2 = gm.ComponentProportion(2);

% syms x
% f1 = weight1 * (1/(sigma1 * sqrt(2 * pi))) * exp(-((x - mu1)^2) / (2 * sigma1^2));
% f2 = weight2 * (1/(sigma2 * sqrt(2 * pi))) * exp(-((x - mu2)^2) / (2 * sigma2^2));
% turning_point = double(vpasolve(f1 == f2, x));
% Define the two Gaussian-like functions (example)
% f1 = @(x) weight1 * (1 / (sigma1 * sqrt(2 * pi))) * exp(-((x - mu1).^2) / (2 * sigma1^2));
% f2 = @(x) weight2 * (1 / (sigma2 * sqrt(2 * pi))) * exp(-((x - mu2).^2) / (2 * sigma2^2));
%
% % Define the difference function
% f_diff = @(x) f1(x) - f2(x);
%
% % Define the initial search range (adjust based on your data)
% x_start = min(flattened_data); % Start of the range
% x_end = max(flattened_data);   % End of the range
%
% % Generate a fine grid of x values for diagnostics
% x_vals = linspace(x_start, x_end, 1000);
% f_vals = arrayfun(f_diff, x_vals);
%
% % Check if a sign change exists in the range
% sign_change_idx = find(diff(sign(f_vals)) ~= 0, 1);
%
% if ~isempty(sign_change_idx)
%     % Refine the interval around the sign change
%     x_start_new = x_vals(sign_change_idx);
%     x_end_new = x_vals(sign_change_idx + 1);
%
%     % Use the refined interval with fzero
%     turning_point = fzero(f_diff, [x_start_new, x_end_new]);
%
%     % Display the result
%     disp(['Turning point found at x = ', num2str(turning_point)]);
% else
%     % If no sign change, minimize the absolute difference
%     disp('No sign change detected. Using fminbnd to find the closest point.');
%     abs_diff = @(x) abs(f1(x) - f2(x));
%     turning_point = fminbnd(abs_diff, x_start, x_end);
%
%     % Display the result
%     disp(['Turning point found by minimizing absolute difference at x = ', num2str(turning_point)]);
% end
%
% % Evaluate the functions at the turning point
% f1_tp = f1(turning_point);
% f2_tp = f2(turning_point);
%
% % Plot the results
% figure;
% histogram(flattened_data,'Normalization','pdf');hold on;
% plot(x_vals, arrayfun(f1, x_vals), 'r', 'LineWidth', 2);
% plot(x_vals, arrayfun(f2, x_vals), 'b', 'LineWidth', 2);
% plot(turning_point, f1_tp, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
% legend('Function 1', 'Function 2', 'Turning Point');
% title('Intersection Detection');
% xlabel('x');
% ylabel('Function Value');
% hold off;
%
% %%

% Define the Gaussian functions
f1 = @(x) weight1 * (1 / (sigma1 * sqrt(2 * pi))) * exp(-((x - mu1)^2) / (2 * sigma1^2));
f2 = @(x) weight2 * (1 / (sigma2 * sqrt(2 * pi))) * exp(-((x - mu2)^2) / (2 * sigma2^2));

% Define the difference function
diff_func = @(x) f1(x) - f2(x);

% Specify the search range (adjust as needed based on your data)
x_start = min(flattened_data); % Start of the range
x_end = max(flattened_data);   % End of the range

% Find the intersection using fzero
turning_point = fzero(diff_func, [x_start, x_end]);

x_vals = linspace(min(flattened_data), max(flattened_data), 1000);
pdf1 = weight1 * normpdf(x_vals, mu1, sigma1);
pdf2 = weight2 * normpdf(x_vals, mu2, sigma2);

figure;
histogram(flattened_data,'Normalization','pdf'); hold on;
plot(x_vals, pdf1, 'r', 'LineWidth', 2);
plot(x_vals, pdf2, 'b', 'LineWidth', 2);
plot(turning_point, double(subs(f1, turning_point)), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
legend('Histogram', 'Gaussian 1', 'Gaussian 2', 'Turning Point');
title('Histogram with Gaussian Fit');
hold off;

%% if using KDE
% Flatten your 260x260 image to a vector
flattened_data = data_temp(:);

% Perform Kernel Density Estimation (KDE) for the data
[pdf_values, x_vals] = ksdensity(flattened_data);

% Find peaks in the KDE estimate
[peak_values, peak_locs] = findpeaks(pdf_values, x_vals);

% Ensure at least two peaks exist
if numel(peak_locs) < 2
    error('KDE found less than two peaks, indicating insufficient clustering.');
end

% Define the KDE-based PDF function
f_kde = @(x) interp1(x_vals, pdf_values, x, 'linear', 0); % Interpolated PDF from KDE

% Find the intersection between the two modes
% Approximate the search range for the intersection (between the peaks)
range_start = min(peak_locs); % Location of the left peak
range_end = max(peak_locs);   % Location of the right peak

% Define the function for intersection (difference between two Gaussian-like peaks)
diff_func = @(x) f_kde(x) - interp1(x_vals, pdf_values, x, 'linear', 0);

% Find the turning point (intersection) within the defined range
turning_point = fzero(diff_func, [range_start, range_end]);

% Plot the results
figure;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); hold on;
plot(turning_point, f_kde(turning_point), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('KDE Estimate', 'Turning Point');
title('KDE and Turning Point Detection');
xlabel('Data Values');
ylabel('Density');
hold off;
%% KDE + Gaussian
% Flatten the image into a vector
flattened_data = data_temp(:);

% Perform KDE on the data
[pdf_values, x_vals] = ksdensity(flattened_data);
% Compute the first derivative (slope)
dx = diff(x_vals); % Spacing in x
dpdf = diff(pdf_values)./ dx; % First derivative (slope)

% Compute the second derivative (curvature)
d2pdf = diff(dpdf)./dx(1:end-1); % Second derivative

% Define thresholds for significant changes
slope_threshold = max(dpdf)* 0.35;  % 40% of maximum slope (adjust as needed)
curvature_threshold = max(abs(d2pdf)) * 0.4; % 10% of maximum curvature

% Find where the slope increases significantly
slope_change_idx = find(dpdf > slope_threshold, 1, 'first');

% Find where curvature shows a significant change
curvature_change_idx = find(abs(d2pdf) > curvature_threshold, 1, 'first');

% Determine the transition point (first significant change)
if ~isempty(slope_change_idx) && ~isempty(curvature_change_idx)
    transition_idx = min(slope_change_idx, curvature_change_idx);
elseif ~isempty(slope_change_idx)
    transition_idx = slope_change_idx;
elseif ~isempty(curvature_change_idx)
    transition_idx = curvature_change_idx;
else
    error('No significant transition point detected.');
end

% Get the x-value of the transition point
transition_point = x_vals(transition_idx);
% Plot the KDE and highlight the transition point
figure;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); hold on;
histogram(flattened_data,'Normalization','pdf');
if ~isempty(transition_point)
    plot(transition_point, interp1(x_vals, pdf_values, transition_point), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
legend('KDE Estimate', 'Transition Point');
title('KDE and Transition Point');
xlabel('Data Values');
ylabel('Density');
hold off;

% Plot derivatives for debugging
figure;
plot(x_vals(1:end-1), dpdf, 'g', 'LineWidth', 1.5); hold on;
if ~isempty(slope_change_idx)
    plot(x_vals(slope_change_idx), dpdf(slope_change_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
title('First Derivative of KDE');
xlabel('Data Values');
ylabel('Slope');
hold off;

figure;
plot(x_vals(1:end-2), d2pdf, 'm', 'LineWidth', 1.5); hold on;
if ~isempty(curvature_change_idx)
    plot(x_vals(curvature_change_idx), d2pdf(curvature_change_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
title('Second Derivative of KDE');
xlabel('Data Values');
ylabel('Curvature');
hold off;

%% for loop detection
% Flatten the image into a vector
flattened_data = data_temp(:);

% Perform Kernel Density Estimation (KDE) for the data
[pdf_values, x_vals] = ksdensity(flattened_data);

% Initialize window size and step
window_size = 10; % Number of points in each segment
step = 5;        % Step size for sliding the window
n_points = numel(x_vals);

% Initialize result storage
p_values = []; % Stores p-values for Gaussianity test
transition_index = NaN; % Stores the transition point

% Slide through the KDE x_vals
for i = 1:step:(n_points - window_size)
    % Extract segment
    segment = x_vals(i:i+window_size);
    segment_pdf = pdf_values(i:i+window_size);

    % Normalize the segment PDF to simulate probabilities
    segment_pdf = segment_pdf / sum(segment_pdf); % Normalize to sum to 1
    replication_factors = round(segment_pdf * 1000); % Scale to integers

    % Ensure replication factors are non-zero
    replication_factors(replication_factors == 0) = 1; % Set a minimum replication factor

    % Replicate the segment values
    segment_data = repelem(segment, replication_factors);

    % Create the x-values for the CDF
    cdf_x_vals = linspace(min(segment_data), max(segment_data), 100);

    % Hypothesized Gaussian CDF values
    cdf_y_vals = normcdf(cdf_x_vals, mean(segment_data), std(segment_data));

    % Create the CDF matrix
    cdf_matrix = [cdf_x_vals', cdf_y_vals'];

    % Perform the KS test
    [h, p] = kstest(segment_data, 'CDF', cdf_matrix);

    % Store p-value
    p_values = [p_values; p];

    % Check for significant deviation
    if h == 1 && isnan(transition_index)
        transition_index = i; % Record first rejection
    end
end

% Determine turning point in x_vals
if ~isnan(transition_index)
    turning_point = x_vals(transition_index);
else
    warning('No clear transition point detected.');
    turning_point = NaN;
end

% Plot the KDE with the transition point
figure;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); hold on;
histogram(projCoV_median_temp(:), 'Normalization', 'pdf'); %hold on;
if ~isnan(turning_point)
    plot(turning_point, interp1(x_vals, pdf_values, turning_point), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
legend('KDE Estimate', 'Transition Point');
title('KDE and Gaussianity Transition Point');
xlabel('Data Values');
ylabel('Density');
hold off;
%% trial 2
% Flatten the image into a vector
flattened_data = projCoV_median_temp(:);

% Perform Kernel Density Estimation (KDE) for the data
[pdf_values, x_vals] = ksdensity(flattened_data);

% Compute the first derivative (slope)
dx = diff(x_vals); % Difference in x (spacing)
dpdf = diff(pdf_values) ./ dx; % First derivative (slope)

% Compute the second derivative (curvature)
d2pdf = diff(dpdf) ./ dx(1:end-1); % Second derivative

% Identify the turning point (end of the platform)
% The turning point is where the slope (dpdf) increases significantly
% OR where the curvature (d2pdf) shows a strong peak

% Thresholds for detecting changes (adjust based on your data)
slope_threshold = 0.01; % Significant slope increase
curvature_threshold = 0.005; % Significant curvature change

% Find indices where slope or curvature exceeds thresholds
slope_change_idx = find(dpdf > slope_threshold, 1, 'first');
curvature_change_idx = find(d2pdf > curvature_threshold, 1, 'first');

% Choose the first detected index as the turning point
if ~isempty(slope_change_idx) && ~isempty(curvature_change_idx)
    turning_idx = min(slope_change_idx, curvature_change_idx);
elseif ~isempty(slope_change_idx)
    turning_idx = slope_change_idx;
elseif ~isempty(curvature_change_idx)
    turning_idx = curvature_change_idx;
else
    warning('No significant turning point detected.');
    turning_idx = NaN;
end

% Determine the turning point value in x_vals
if ~isnan(turning_idx)
    turning_point = x_vals(turning_idx);
else
    turning_point = NaN;
end

% Plot the results
figure;
plot(x_vals, pdf_values, 'b', 'LineWidth', 2); hold on;
if ~isnan(turning_point)
    plot(turning_point, interp1(x_vals, pdf_values, turning_point), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
legend('KDE Estimate', 'Turning Point');
title('KDE and Turning Point Detection');
xlabel('Data Values');
ylabel('Density');
hold off;

% Optionally, plot the derivatives for debugging
figure;
plot(x_vals(1:end-1), dpdf, 'g', 'LineWidth', 1.5); hold on;
if ~isnan(slope_change_idx)
    plot(x_vals(slope_change_idx), dpdf(slope_change_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
title('First Derivative of KDE');
xlabel('Data Values');
ylabel('Slope (First Derivative)');
hold off;

figure;
plot(x_vals(1:end-2), d2pdf, 'm', 'LineWidth', 1.5); hold on;
if ~isnan(curvature_change_idx)
    plot(x_vals(curvature_change_idx), d2pdf(curvature_change_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
title('Second Derivative of KDE');
xlabel('Data Values');
ylabel('Curvature (Second Derivative)');
hold off;


