function convert_to_nifti(app)


save_nifti = fullfile(app.savefns,'nifti');
if ~exist(save_nifti,'dir')
    mkdir(save_nifti)
end
    
%%% read file
save_mask_fn = fullfile(app.save_mat,'avg_mask_all.mat'); 
save_retina_fn_Z = fullfile(app.save_mat,strcat('registered_',app.tp,'_enfaceOCTA_fromZeiss.mat'));

niftiFile1 = importdata(save_mask_fn);
retina_reg_Z = importdata(save_retina_fn_Z);

retina_reg_Z = retina_reg_Z(app.cropped_amount+1:end-app.cropped_amount,app.cropped_amount+1:end-app.cropped_amount,:);

niftiwrite(niftiFile1, fullfile(save_nifti,'avg_mask_all.nii'))
niftiwrite(mat2gray(mean(retina_reg_Z,3)), fullfile(save_nifti,strcat('registered_',app.tp,'_enfaceOCTA_fromZeiss.nii')));
end









