function cov_visualization(app)

outPut(app, ["Final CoV projection images are saved in: ", app.save_comp]);

% initialize variables
all_cov_mean = cell(length(app.timepoints),1);
all_cov_median = cell(length(app.timepoints),1);

% all_cov_mean_hist = cell(length(app.timepoints),1);
% all_cov_median_hist = cell(length(app.timepoints),1);

all_cov_fromV = cell(length(app.timepoints),1);
all_cov_fromZ = cell(length(app.timepoints),1);
%%
for m = 1:length(app.timepoints)
    app.tp = app.timepoints{m};
    cm = getcolormap_sup();
    % create all saving folders
    app.savefns = fullfile(app.save_folder, app.tp);%,'Extraction');
    app.save_cov = fullfile(app.savefns,'CoV');
    app.save_mat = fullfile(app.savefns,'mat');
    app.save_mat_proj = fullfile(app.save_mat,app.proj_layer);
    app.save_cov_proj = fullfile(app.save_cov,app.proj_layer);

    % save_csv = fullfile(savefns,'csv');
    % if ~exist(save_csv)
    %     mkdir(save_csv)
    % end 

    save_nifti = fullfile(app.savefns,'nifti');

    noiseCorr_offset = importdata(fullfile(app.save_mat,'offset.mat'));

    % save_projOCTA_fn = fullfile(save_cov_proj,strcat('Binarized_avgOCTA_proj_offset_',strrep(num2str(noiseCorr_offset),'.','-'),'.mat'));
    save_covMedian_fn = fullfile(app.save_mat_proj,strcat('CoV_proj_fromBscans_median_offset_',num2str(noiseCorr_offset),'.mat'));
    
    save_covMean_fn = fullfile(app.save_mat_proj,strcat('CoV_proj_fromBscans_mean_offset_',num2str(noiseCorr_offset),'.mat'));
    
    save_mask_fn = fullfile(save_nifti,'avg_mask_all_corr.nii');
    if ~exist(save_mask_fn,'file')
        save_mask_fn = fullfile(save_nifti,'avg_mask_all.nii');
    end
    
    % save_mask_fn = fullfile(app.save_mat,'avg_mask_all.mat')

    savefn_mat_V = fullfile(app.save_mat_proj,strcat(app.tp,'_OCTA_fromVolume_CoV','.mat'));
    savefn_mat_Z = fullfile(app.save_mat_proj,strcat(app.tp,'_OCTA_fromZeiss_CoV','.mat'));

    % save_normcovMedian_fn = fullfile(save_mat_proj,strcat('Normalized_projCoV_fromBscans_median_offset_',num2str(noiseCorr_offset),'.mat'));
    % save_normcovMean_fn = fullfile(save_mat_proj,strcat('Normalized_projCoV_fromBscans_mean_offset_',num2str(noiseCorr_offset),'.mat'));
    % 
    % covBscan_median_norm = mat2rgay(covBscan_median);
    % covBscan_mean_norm = mat2gray(covBscan_mean);
    % 
    % % save(save_normcovMedian_fn,'covBscan_median','-v7.3');
    % % save(save_normcovMean_fn,'covBscan_mean','-v7.3');

    %%% read CoV map and draw historgam
    % avg_mask_all = importdata(save_mask_fn);
    avg_mask_all = double(niftiread(save_mask_fn));
    % avg_OCTA_proj = importdata(save_projOCTA_fn);
    covBscan_mean = importdata(save_covMean_fn);
    covBscan_median = importdata(save_covMedian_fn);

    % imshow((covBscan_mean));colormap(cm)

    cov_fromV = importdata(savefn_mat_V);
    cov_fromZ = importdata(savefn_mat_Z);

    covBscan_mean(isnan(covBscan_mean)) = 0;
    covBscan_median(isnan(covBscan_median)) = 0;

    cov_fromV(isnan(cov_fromV)) = 0;
    cov_fromZ(isnan(cov_fromZ)) = 0;

    all_cov_mean{m} = mat2gray(covBscan_mean.*avg_mask_all);
    all_cov_median{m} = mat2gray(covBscan_median.*avg_mask_all);

    all_cov_fromV{m} =  mat2gray(cov_fromV.*avg_mask_all);
    all_cov_fromZ{m} =  mat2gray(cov_fromZ.*avg_mask_all);

    %%% choose the threshold value based on histogram
    % projOCTA_mask = avg_OCTA_proj.*avg_mask_all;
    % covBscan_AVG_mask = covBscan_mean.*avg_mask_all;
    covBscan_MED_mask = covBscan_median.*avg_mask_all;

    cov_fromV_mask = cov_fromV.*avg_mask_all;
    cov_fromZ_mask = cov_fromZ.*avg_mask_all;

    %%% compute mean and std
    measurement = struct;
    measurement.mu_customV = mean(nonzeros(covBscan_MED_mask));
    measurement.std_customV = std(nonzeros(covBscan_MED_mask));
    % 
    % list_customV = reshape(nonzeros(covBscan_MED_mask),1,[]);   
    % [counts_customV,edges_customV]=histcounts(list_customV,'Normalization','pdf','NumBins',100);
    % esti_list_customV = normpdf(edges_customV, mean(nonzeros(covBscan_MED_mask)),std(nonzeros(covBscan_MED_mask)));
    % figure;
    % plot(esti_list_customV);hold on
    % plot(counts_customV);hold off
    % err_customV = sum(abs(counts_customV - esti_list_customV(1:100)))/sum(esti_list_customV(1:100))
    % 
    measurement.mu_customE = mean(nonzeros(cov_fromV_mask));
    measurement.std_customE = std(nonzeros(cov_fromV_mask));
    % 
    % list_customE = reshape(nonzeros(cov_fromV_mask),1,[]);
    % [counts_customE,edges_customE]=histcounts(list_customE,'Normalization','pdf','NumBins',100);
    % esti_list_customE = normpdf(edges_customE, mean(nonzeros(cov_fromV_mask)),std(nonzeros(cov_fromV_mask)));
    % figure;
    % plot(esti_list_customE);hold on
    % plot(counts_customE);hold off%normrnd(mean(list_zeissE),std(list_zeissE),[10000,1]);
    % err_customE = sum(abs(counts_customE - esti_list_customE(1:100)))/sum(esti_list_customE(1:100))
    % 
    % 
    measurement.mu_zeissE = mean(nonzeros(cov_fromZ_mask));
    measurement.std_zeissE = std(nonzeros(cov_fromZ_mask));

    % 
    % list_zeissE = reshape(nonzeros(cov_fromZ_mask),1,[]);
    % [counts_zeissE,edges_zeissE]=histcounts(list_zeissE,'Normalization','pdf','NumBins',100);
    % esti_list_zeissE = normpdf(edges_zeissE, mean(nonzeros(cov_fromZ_mask)),std(nonzeros(cov_fromZ_mask)));
    % figure;
    % plot(esti_list_zeissE);hold on
    % plot(counts_zeissE);hold off
    % err_zeissE = sum(abs(counts_zeissE - esti_list_zeissE(1:100)))/sum(esti_list_zeissE(1:100))

    % figure;
    % plot(esti_list_zeissE);hold on
    % plot(esti_list_customE);
    % figure;
    % histogram(list_zeissE,'Normalization','pdf','NumBins',100);hold on
    % histogram(esti_list_zeissE,'Normalization','pdf','NumBins',100);hold off


    %%% Compute the negative log
    % pd_customV = fitdist(nonzeros(covBscan_MED_mask),'Normal');
    % pd_customV.NLogL
    % 
    % pd_customE = fitdist(nonzeros(cov_fromV_mask),'Normal');
    % pd_customE.NLogL
    % 
    % pd_zeissE = fitdist(nonzeros(cov_fromZ_mask),'Normal');
    % pd_zeissE.NLogL

    th = struct;

    %%% Plots & threshold
    %%% CoV from Zeiss
    close all
    % fZ_hist = figure;
    figure('Visible','off');
    histogram(nonzeros(cov_fromZ_mask),'NumBins',150,'Normalization','probability');
    title('CoV values from instrument-processed data','FontName','Times New Roman','FontSize',12); 
    % ylabel('Normalized # of Pixels');xlabel('CoV Value [a.u.]')
    xlabel('Pixel Values [a.u.]','FontName','Times New Roman')
    ylabel('Normalized # of Pixels','FontName','Times New Roman')
    xlim([0,1])
    % ylim([0,1200])
    fZ_hist = gca;
    xt = get(fZ_hist, 'YTick');
    set(fZ_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_CoV_fromZeiss_histogram.png'));
    exportgraphics(fZ_hist,savefm_hist)
    close all

    %%% normalize CoV
    % cov_fromV_mask_norm = cov_fromV_mask - min(cov_fromV_mask(:));
    % cov_fromV_mask_norm = cov_fromV_mask_norm./max(cov_fromV_mask_norm(:));

    % cov_fromZ_mask_norm = cov_fromZ_mask; % mat2gray(cov_fromZ_mask);
    % clear cov_fromZ_mask
    % 
    % figure('Visible','off');
    % histogram(nonzeros(cov_fromZ_mask_norm),'NumBins',150,'Normalization','probability');
    % title('Normalized CoV values from instrument-processed data','FontName','Times New Roman','FontSize',12); 
    % % ylabel('Normalized # of Pixels');xlabel('CoV Value [a.u.]')
    % xlabel('Pixel Values [a.u.]','FontName','Times New Roman')
    % ylabel('Normalized # of Pixels','FontName','Times New Roman')
    % xlim([0,1])
    % % ylim([0,1200])
    % fZ_hist = gca;
    % xt = get(fZ_hist, 'YTick');
    % set(fZ_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    % savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_CoV_fromZeiss_histogram.png'));
    % exportgraphics(fZ_hist,savefm_hist)
    % close(fZ_hist)

    ptitle = 'CoV values from instrument-processed data';
    q = strcat(app.tp," - what is the threshold value: ");
    threshold_Z = inputDialogWithPlot(cov_fromZ_mask, ptitle, q); %input("What is the threshold value: ");
    outPut(app,['Threshold value (Zeiss_E) = ', num2str(threshold_Z)])

    cov_fromZ_mask(cov_fromZ_mask > threshold_Z) = threshold_Z;

    th.threshold_Z = threshold_Z;

    fZ = figure('Visible','off');
    imshow(ind2rgb(round(mat2gray(cov_fromZ_mask).*255),cm),cm);% mat2gray
    % title("CoV from Zeiss");cbar_mean = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0',num2str(threshold_Z)});
    savefn_img = fullfile(app.save_comp,strcat(app.tp,'_normCoV_fromZeiss_threshold_',strrep(num2str(threshold_Z),'.','-'),'.png'));
    exportgraphics(fZ,savefn_img)
    close all

    % CoV from volume
    % close all
    % fV_hist = figure;
    figure('Visible','off');
    histogram(nonzeros(cov_fromV_mask),'NumBins',150,'Normalization','probability');% 'Normalization','count'
    title('CoV values from volume-extracted data','FontName','Times New Roman','FontSize',12);% 
    ylabel('Normalized # of Pixels','FontName','Times New Roman');xlabel('CoV Value [a.u.]','FontName','Times New Roman')
    xlim([0,1])
    % ylim([0,1200])
    fV_hist = gca;
    xt = get(fV_hist, 'YTick');
    set(fV_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    savefm_hist = fullfile(app.save_comp, strcat(app.tp,'_CoV_fromVolume_histogram.png'));
    exportgraphics(fV_hist,savefm_hist)
    close all

    %%% normalize CoV
    % cov_fromV_mask_norm = cov_fromV_mask - min(cov_fromV_mask(:));
    % cov_fromV_mask_norm = cov_fromV_mask_norm./max(cov_fromV_mask_norm(:));

    % cov_fromV_mask_norm = mat2gray(cov_fromV_mask);
    % clear cov_fromV_mask
    % 
    % figure('Visible','off');
    % histogram(nonzeros(cov_fromV_mask_norm),'NumBins',150,'Normalization','probability');
    % title('Normalized CoV values from volume-extracted data','FontName','Times New Roman','FontSize',12);% 
    % ylabel('Normalized # of Pixels','FontName','Times New Roman');xlabel('CoV Value [a.u.]','FontName','Times New Roman')
    % xlim([0,1])
    % % ylim([0,1200])
    % fV_hist = gca;
    % xt = get(fV_hist, 'YTick');
    % set(fV_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    % savefm_hist = fullfile(app.save_comp, strcat(app.tp,'_normCoV_fromVolume_histogram.png'))
    % exportgraphics(fV_hist,savefm_hist)


    ptitle = 'CoV values from volume-extracted data';
  
    threshold_V = inputDialogWithPlot(cov_fromV_mask, ptitle, q); %input("What is the threshold value: ");

    outPut(app,['Threshold value (Custom_E) = ', num2str(threshold_V)])
    cov_fromV_mask(cov_fromV_mask > threshold_V) = threshold_V;

    th.threshold_V = threshold_V;

    fV = figure('Visible','off');
    imshow(ind2rgb(round(mat2gray(cov_fromV_mask).*255),cm),cm);% ? mat2gray or...? 
    % %title("CoV from Volume");cbar_mean = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0',num2str(threshold_V)});
    savefn_img = fullfile(app.save_comp,strcat(app.tp,'_normCoV_fromVolume_threshold_',strrep(num2str(threshold_V),'.','-'),'.png'));
    exportgraphics(fV,savefn_img)

    %%% mean CoV from Bscans
    % % close all
    % % fmean_hist = figure;
    % figure;
    % histogram(nonzeros(covBscan_AVG_mask),'NumBins',150,'Normalization','probability');
    % % title('Histogram of CoV values from B-scan-based CoV computation','FontName','Times New Roman');%
    % xlabel('Pixel Values [a.u.]')
    % ylabel('Normalized # of Pixels')
    % xlim([0,1])
    % % ylim([0,1200])
    % fmean_hist = gca;
    % xt = get(fmean_hist, 'YTick');
    % set(fmean_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    % savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_meanCoV_histogram.png'));
    % exportgraphics(fmean_hist,savefm_hist)
    % % close
    % 
    % %%% Normalize CoV
    % covBscan_AVG_mask_norm = mat2gray(covBscan_AVG_mask);
    % save(save_normcovMean_fn,'covBscan_AVG_mask_norm','-v7.3')
    % 
    % figure;
    % histogram(nonzeros(covBscan_AVG_mask_norm),'NumBins',150,'Normalization','probability');
    % %title('Histogram of Mean Projection of Volumetric CoV');%
    % xlabel('Pixel Values [a.u.]')
    % ylabel('Normalized # of Pixels')
    % % xlim([0,0.8])
    % % ylim([0,1200])
    % fmean_hist = gca;
    % xt = get(fmean_hist, 'YTick');
    % set(fmean_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    % savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_meanCoV_norm_histogram.png'));
    % exportgraphics(fmean_hist,savefm_hist)
    % 
    % threshold_Mean = input("What is the threshold value: ");
    % covBscan_AVG_mask_norm(covBscan_AVG_mask_norm > threshold_Mean) = threshold_Mean;
    % 
    % fmean = figure;
    % imshow(ind2rgb(round((covBscan_AVG_mask_norm).*255),cm),cm);
    % title("CoV of Mean OCTA Bscan",'FontName','Times New Roman');%cbar_mean = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0',num2str(threshold_Mean)});
    % savefn_img = fullfile(app.save_comp,strcat(app.tp,'_normCoV_proj_fromVolume_mean_threshold_',strrep(num2str(threshold_Mean),'.','-'),'_no_title.png'));
    % exportgraphics(fmean,savefn_img)
    % 
    % %%% differenciate vessels at different layers
    % covBscan_AVG_sup = covBscan_AVG_mask_norm;
    % deep_AVG_threshold = input("What is the threshold value for deep vessels: ");
    % covBscan_AVG_sup(covBscan_AVG_sup > (deep_AVG_threshold)) = 0;
    % 
    % fmean_sup = figure;
    % imshow(ind2rgb(round((covBscan_AVG_sup).*255),cm),cm);
    % %title(["Mean CoV of Superficial Vessel",...
    %     %"from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_AVG_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_AVG_threshold/threshold_Mean,2)),'1'});
    % ax = gca;
    % %ax.TitleFontSizeMultiplier = 0.8;
    % savefn_img = fullfile(app.save_comp,strcat(app.tp,'_supVessel_CoV_proj_fromVolume_mean_threshold_',strrep(num2str(deep_AVG_threshold),'.','-'),'_no_title.png'));
    % exportgraphics(fmean_sup,savefn_img)
    % 
    % covBscan_AVG_deep = covBscan_AVG_mask_norm;
    % covBscan_AVG_deep(covBscan_AVG_deep <= (deep_AVG_threshold)) = 0;
    % 
    % fmean_deep = figure;
    % imshow(ind2rgb(round((covBscan_AVG_deep).*255),cm),cm);
    % % title(["Mean CoV of Deep Vessel",...
    % %     "from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_AVG_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_AVG_threshold/threshold_Mean,2)),'1'});
    % % cbar_mean = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0','1'},'Location','southoutside');
    % ax = gca;
    % ax.TitleFontSizeMultiplier = 0.8;
    % savefn_img = fullfile(app.save_comp,strcat(app.tp,'_deepVessel_CoV_proj_fromVolume_mean_threshold_',strrep(num2str(deep_AVG_threshold),'.','-'),'_no_title.png'));
    % exportgraphics(fmean_deep,savefn_img)

    %%% Median CoV from Bscans
    % close all
    % fmedian_hist = figure;
    figure('Visible','off');
    histogram(nonzeros(covBscan_MED_mask),'NumBins',150,'Normalization','probability');
    title('CoV values from B-scan basis CoV computation','FontName','Times New Roman','FontSize',12);%'
    xlabel('Pixel Values [a.u.]','FontName','Times New Roman')
    ylabel('Normalized # of Pixels','FontName','Times New Roman')
    xlim([0,1])
    % ylim([0,1200])
    fmedian_hist = gca;
    xt = get(fmedian_hist, 'YTick');
    set(fmedian_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_medianCoV_histogram.png'));
    exportgraphics(fmedian_hist,savefm_hist)
    close all

    %%% normalization
    % covBscan_MED_mask_norm = covBscan_MED_mask; % mat2gray(covBscan_MED_mask);
    % clear covBscan_MED_mask
    % save(save_normcovMedian_fn,'covBscan_MED_mask_norm','-v7.3')

    % figure('Visible','off'); 
    % histogram(nonzeros(covBscan_MED_mask),'NumBins',150,'Normalization','probability');
    % title('Normalized CoV values from B-scan basis CoV computation','FontName','Times New Roman','FontSize',12);%'
    % xlabel('Pixel Values [a.u.]','FontName','Times New Roman')
    % ylabel('Normalized # of Pixels','FontName','Times New Roman')
    % xlim([0,1])
    % % ylim([0,1200])
    % fmedian_hist = gca;
    % xt = get(fmedian_hist, 'YTick');
    % set(fmedian_hist, 'YTick', xt, 'YTickLabel', round(xt/max(xt),1))
    % savefm_hist =  fullfile(app.save_comp, strcat(app.tp,'_medianCoV_histogram.png'));
    % exportgraphics(fmedian_hist,savefm_hist)
    % close(fmedian_hist)

    ptitle = 'CoV values from B-scan basis CoV computation';
    threshold_Median = inputDialogWithPlot(covBscan_MED_mask, ptitle, q);% input("What is the threshold value: ");
    outPut(app,['Threshold value (Custom_V) = ', num2str(threshold_Median)])

    
    covBscan_MED_mask(covBscan_MED_mask > threshold_Median) = threshold_Median; % covBscan_MED_mask_norm

    th.threshold_Median = threshold_Median;

    fmedian = figure('Visible','off');
    savefn_img = fullfile(app.save_comp,strcat(app.tp,'_normCoV_proj_fromVolume_median_threshold_',strrep(num2str(threshold_Median),'.','-'),'.png'));
    imshow(ind2rgb(round(mat2gray(covBscan_MED_mask).*255),cm),cm);% mat2gray  
    % title("CoV of Median OCTA Bscan");cbar_median = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0',num2str(threshold_Median)});
    exportgraphics(fmedian,savefn_img)
    close all

    %%% differenciate vessels at different layers
    covBscan_MED_sup = mat2gray(covBscan_MED_mask);

    q = strcat(app.tp," - what is the threshold value to separate the deep vessels: ");
    deep_MED_threshold = findTurningPointLognorm(app, covBscan_MED_mask, ptitle, q);% inputdlg("What is the threshold value for deep vessels: ");
    outPut(app,['Threshold value to separate the deep vessel = ', num2str(deep_MED_threshold)]);
    th.deep_MED_threshold = deep_MED_threshold;

    covBscan_MED_sup(covBscan_MED_sup > deep_MED_threshold) = 0; % (deep_MED_threshold/threshold_Median) if normalized

    th.deep_MED_threshold = deep_MED_threshold;
    measurement.mu_customeV_sup = mean(nonzeros(covBscan_MED_sup));
    measurement.std_customeV_sup = std(nonzeros(covBscan_MED_sup));
    % mu_customeV_sup = mean(nonzeros(covBscan_MED_sup))
    % std_customeV_sup = std(nonzeros(covBscan_MED_sup))


    fmedian_sup = figure('Visible','off');
    imshow(ind2rgb(round((covBscan_MED_sup).*255),cm),cm); % mat2gray
    % title(["Median CoV of Superficial Vessel",...
    %     "from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_MED_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_MED_threshold/threshold_Mean,2)),'1'});
    ax = gca;
    ax.TitleFontSizeMultiplier = 0.8;
    savefn_img = fullfile(app.save_comp,strcat(app.tp,'_supVessel_normCoV_proj_fromVolume_median_threshold_',strrep(num2str(deep_MED_threshold),'.','-'),'.png'));
    exportgraphics(fmedian_sup,savefn_img)
    close all

    covBscan_MED_deep = covBscan_MED_mask;
    covBscan_MED_deep(covBscan_MED_deep <= deep_MED_threshold) = 0; % (deep_MED_threshold/threshold_Median) if mat2gray

    measurement.mu_customeV_deep = mean(nonzeros(covBscan_MED_deep));
    measurement.std_customeV_deep = std(nonzeros(covBscan_MED_deep));
    % mu_customeV_deep = mean(nonzeros(covBscan_MED_deep))
    % std_customeV_deep = std(nonzeros(covBscan_MED_deep))

    fmedian_deep = figure('Visible','off');
    imshow(ind2rgb(round((covBscan_MED_deep).*255),cm),cm); % mat2gray
    % title(["Median CoV of Deep Vessel",...
    %     "from OCTA Bscan"]);cbar_mean = colorbar('eastoutside','Ticks',[0,deep_MED_threshold/threshold_Mean,1],'TickLabels',{'0',num2str(round(deep_MED_threshold/threshold_Mean,2)),'1'});
    ax = gca;
    ax.TitleFontSizeMultiplier = 0.8;
    savefn_img = fullfile(app.save_comp,strcat(app.tp,'_deepVessel_normCoV_proj_fromVolume_median_threshold_',strrep(num2str(deep_MED_threshold),'.','-'),'.png'));
    exportgraphics(fmedian_deep,savefn_img)
    close all

    save(fullfile(app.save_comp, strcat(app.tp,'_userinput_threshold.mat')),'th','-v7.3');
    save(fullfile(app.save_comp, strcat(app.tp,'_measurement.mat')),"measurement",'-v7.3');

    % %%% Combined plots
    % cov_vis = figure;
    % subplot(1,3,1);imshow(mat2gray(projOCTA_mask));title("Binarized En Face OCTA");cbar = colorbar('eastoutside','Ticks',[0,1],'TickLabels',{'0','1'});
    % subplot(1,3,2);imshow(ind2rgb(round((covBscan_AVG_mask_norm).*255),cm),cm);title("Normalized CoV of Mean OCTA Bscan");cbar_mean = colorbar('eastoutside','Ticks',[0,deep_AVG_threshold,1],'TickLabels',{'0',num2str(round(deep_AVG_threshold,2)),'1'});
    % subplot(1,3,3);imshow(ind2rgb(round((covBscan_MED_mask_norm).*255),cm),cm);title("Normalized CoV of Median OCTA Bscan");cbar_median = colorbar('eastoutside','Ticks',[0,deep_MED_threshold,1],'TickLabels',{'0',num2str(round(deep_MED_threshold,2)),'1'});
    % set(gcf,'position',get(0,'screensize'));
    % save_img_fn = fullfile(app.save_comp, strcat(app.tp,'_compareCoV_threshold_',strrep(num2str(threshold_Mean),'.','-'),'_',strrep(num2str(threshold_Median),'.','-'),'.png'));
    % exportgraphics(cov_vis, save_img_fn)
end

end



