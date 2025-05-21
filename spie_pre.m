

root = 'F:\00_CoV\CONTROL\CV003\Results3D\OD_MAC_12190100PM\mat';
segmentation = read_DNN_Segmentation('F:\00_CoV\CONTROL\CV003\CV003_CUBE\OD_MAC_12190100PM\CropSegmentation\CV003_OD_MAC_12190100PM_Angiography_3x3_cube_z_all\Boundary.nii.gz');

depthROI_CoV = importdata('F:\00_CoV\CONTROL\CV003\Results3D\OD_MAC_12190100PM\mat\depthROI_CoV.mat');
cropped_amount = 20;

segmentation = segmentation(depthROI_CoV(1):depthROI_CoV(2),cropped_amount+1:end-cropped_amount,cropped_amount+1:end-cropped_amount);

segILM = segmentation;
segILM(~(segILM == 1))= 0;
% segILM((segILM == 2))= 1;

segNFL = segmentation;
segNFL(~(segNFL == 2))= 0;
segNFL((segNFL == 2))= 1;

segONL = segmentation;
segONL(~(segONL == 4))= 0;
segONL(segONL == 4)= 1;



%%
vol_check = importdata(fullfile(root,'frame_156_fromVolume.mat'));

[row_NFL, ~] = layer_correction(segNFL, 156); % Finding the segmentation 1 value locations
[row_ONL, ~] = layer_correction(segONL, 156); % Finding the segmentation 1 value locations

row_NFL_smooth= row_NFL;
row_ONL_smooth = row_ONL;
jj= 112;

for i = 1:size(vol_check,3)
    plot(vol_check(row_NFL_smooth(jj):row_ONL_smooth(jj), jj, i));hold on
    [~,max_idx(i)] = max(vol_check(row_NFL_smooth(jj):row_ONL_smooth(jj), jj, i));
end

figure;plot(1:1:size(vol_check,3),row_NFL_smooth(jj)+max_idx);
xlabel('Number of Volume');
ylabel('Depth Index')
ylim([row_NFL_smooth(jj),row_ONL_smooth(jj)])

%% evaluate the B-scan basis CoV
cov_vol_binarized_ref = importdata('F:\00_CoV\CONTROL\CV003\Results3D\OD_MAC_12190100PM\mat\Ref_Binarized_CoV_fromOCTA_Bscans_offset_10.mat');

plot(cov_vol_binarized_ref(row_NFL_smooth(jj):row_ONL_smooth(jj), jj));hold on


%% Example data (remove or replace with your own data)
cov_Bscan_ref = cov_vol_binarized_ref(row_NFL_smooth(jj):row_ONL_smooth(jj), jj);
% Plot the data
figure;
plot(cov_Bscan_ref);
hold on;

% Calculate the median
medianVal = median(nonzeros(cov_Bscan_ref));

% Plot the median as a horizontal line
plot([1, length(cov_Bscan_ref)], [medianVal, medianVal], 'r--', 'LineWidth', 2);

% Labeling and legend
xlabel('Index');
ylabel('Value');
title('Ref\_binarized\_CoV with Median');
legend('Data','Median','Location','best');
grid on;


%%
figure;
imshow(cov_vol_binarized_ref(row_NFL_smooth(jj):row_ONL_smooth(jj),:),[]);hold on
xline(112, 'r', 'LineWidth', 1,'LineStyle','--');

figure;
% Plot the median as a horizontal line
plot(cov_Bscan_ref, row_NFL_smooth(jj):row_ONL_smooth(jj));hold on;
plot( [medianVal], row_NFL_smooth(jj):row_ONL_smooth(jj),'r--', 'LineWidth', 2);

%% generate the en face image of maximum value index
avg_mask_all = double(niftiread('F:\00_CoV\CONTROL\CV003\Results3D\OD_MAC_12190100PM\nifti\avg_mask_all_corr.nii'));
maxValIdx_CoV = importdata('F:\00_CoV\CONTROL\CV003\Results3D\OD_MAC_12190100PM\mat\NFL_ONL\OD_MAC_12190100PM_maxValIdx_fromVolume_CoV.mat');
maxValIdx_CoV_masked = maxValIdx_CoV.*avg_mask_all;

