function compute_CoV_2D(reg_proj_stack, save_mat_fn)

% reg_proj_stack = reg_proj_stack(cropped_amount+1:end-cropped_amount,cropped_amount+1:end-cropped_amount,:);

reg_proj_mean = mean(reg_proj_stack, 3, "omitnan");
reg_proj_mean(isnan(reg_proj_mean)) = 0;

% UPSAMPLING
% reg_proj_mean = imresize(reg_proj_mean,[1024 1024]);
% reg_proj_stack = imresize(reg_proj_stack,[1024 1024]);

% final_masked_proj = reg_proj_stack;

% saveAs3DTiffStack(final_masked_proj,fullfile(save_cov,strcat('masked_',timepoint,'_enfaceOCTA_fromVolume.tif')));
% saveAs3DTiffStack(final_masked_retina_V,fullfile(save_cov,strcat('masked_',timepoint,'_enfaceOCTA_fromZeiss.tif')));

%%% CoV calculation
final_proj_CoV_ini = zeros(size(reg_proj_mean));

tic
for ii = 1:size(reg_proj_stack,1)
    for jj = 1:size(reg_proj_stack,2)
        final_proj_CoV_ini(ii,jj) = getCV(squeeze(reg_proj_stack(ii,jj,:)));
    end
end

for idx = 1:size(reg_proj_stack,3)
    imshow(mat2gray(reg_proj_stack(:,:,idx)));pause(0.1)
end


save(save_mat_fn,'final_proj_CoV_ini');
toc

% %%% Visualization
% cm = getcolormap_sup();
% 
% % figure;
% final_proj_CoV = final_proj_CoV_ini;
% final_proj_CoV(isnan(final_proj_CoV)) = 0;
% 
% % check the histogram
% h = figure;
% histogram(final_proj_CoV);xlabel('Pixel Value');ylabel('Number of Pixels')
% save_hist_fn = strrep(save_img_fn,'_CoV.png','_histogram.png');
% exportgraphics(h, save_hist_fn);
% 
% prompt = "What is the threshold value? ";
% threshold = input(prompt);
% 
% %  Use threshold
% final_proj_CoV_masked = final_proj_CoV.*avg_mask_all;
% final_proj_CoV_masked(final_proj_CoV_masked > threshold) = threshold;
% 
% %  Use normalization
% % final_proj_CoV_norm = final_proj_CoV - min(final_proj_CoV(:));
% % final_proj_CoV_norm = final_proj_CoV_norm./max(final_proj_CoV_norm(:));
% 
% % final_retina_CoV_norm = final_retina_CoV_norm;
% 
% final_proj_CoV_norm = round(mat2gray(final_proj_CoV_masked).*255);
% final_proj_CoV_norm = ind2rgb(final_proj_CoV_norm, cm);
% figure;imshow(final_proj_CoV_norm,cm); colorbar('Ticks', [0,1],'TickLabels',{'0',num2str(threshold)});
% 
% x = gcf;
% exportgraphics(x, save_img_fn)

close all
end