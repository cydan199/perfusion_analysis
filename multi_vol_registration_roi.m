function multi_vol_registration_roi(app, ref_enface, ref_vol, depthROI)
app.CancelFlag = false;

fig = app.UIFigure;
d = uiprogressdlg(fig,'Title','Multi-volume Registration',...
        'Indeterminate','on','Cancelable','on');
drawnow

numPoints = size(ref_vol(:,:,:),1);%datasize(2);
numBscans = size(ref_vol(:,:,:),3);%datasize(3);
numAscans = size(ref_vol(:,:,:),2);%datasize(1);

d.Message = "Creating saving folders...";

save_reg_proj = fullfile(app.savefns,'proj_fromReg');
if ~exist(save_reg_proj,'dir')
    mkdir(save_reg_proj)
end

% use this save folder only when stack of hyalocyte is generated
save_OCT_registered_V = fullfile(save_reg_proj,'Registered_V');
if ~exist(save_OCT_registered_V,'dir')
    mkdir(save_OCT_registered_V)
end
% 
save_OCTA_registered_Z = fullfile(save_reg_proj,'Registered_Z');
if ~exist(save_OCTA_registered_Z,'dir')
    mkdir(save_OCTA_registered_Z)
end

d.Message = "Loading data...";

retina_fns_not_ordered = dir(fullfile(app.loadloc_enface,'*Retina*.bmp'));
retina_fns = natsortfiles(retina_fns_not_ordered); clear retina_fns_not_ordered

OCT_vol_fns_not_ordered = dir(fullfile(app.loadloc_cube,'*_cube_z*.img'));
OCT_vol_fns = natsortfiles(OCT_vol_fns_not_ordered); clear OCT_vol_fns_not_ordered

if d.CancelRequested == 1
    app.CancelFlag = true;
    return 
end

retina_reg_Z = zeros(numAscans,numBscans,length(retina_fns));
all_OCT = zeros(app.datasize(2),app.datasize(1),app.datasize(3));
all_OCTA = zeros(app.datasize(2),app.datasize(1),app.datasize(3));

for ind = 1:length(retina_fns)
    % disp(num2str(ind))

    msg = strcat("Registering"," ",num2str(ind)," ","out of ",num2str(length(retina_fns))," volumes at timepoint"," ",app.tp);
    d.Message = msg;

    d.Message = [msg, strcat("en face image: ",retina_fns(ind).name)];
    msg = [msg, strcat("en face image: ",retina_fns(ind).name)];
    outPut(app, ["en face image: ",retina_fns(ind).name]);

    retina = double(imread(fullfile(retina_fns(ind).folder, retina_fns(ind).name)));
    resize_retina = flipud(permute(imresize(retina,[numAscans numBscans],'bilinear'),[2,1]));
    retina = resize_retina; clear resize_retina;

    OCTA_vol_fn = strrep(OCT_vol_fns(ind).name,'cube_z','FlowCube_z');

    savefn_projOCT_V = fullfile(fullfile(save_OCT_registered_V,retina_fns(ind).name));

    savefnIMG_OCT = fullfile(fullfile(OCT_vol_fns(ind).folder,'CropRegistered'), OCT_vol_fns(ind).name);
    savefnTIF_OCT = strrep(savefnIMG_OCT,'.img', '.tif');
    
    savefnIMG_OCTA = fullfile(fullfile(OCT_vol_fns(ind).folder,'CropRegistered'), OCTA_vol_fn);
    savefnTIF_OCTA = strrep(savefnIMG_OCTA,'.img', '.tif');

    % Rigid registration
    d.Message = [msg,"Rigid registration"];

    [tformTotal, OutputView] = BRISK_SURF_ZPE((mat2gray(ref_enface)), (mat2gray(retina)));
    retina_reg = imwarp(retina, tformTotal, 'OutputView', OutputView);

    d.Message = [msg, strcat("OCT volume: ",OCT_vol_fns(ind).name)];
    msg = [msg, strcat("OCT volume: ",OCT_vol_fns(ind).name)];
    outPut(app, ["OCT volume: ",OCT_vol_fns(ind).name])

    OCT_vol = double(read_OCT_vol(app, fullfile(OCT_vol_fns(ind).folder, OCT_vol_fns(ind).name)));
    if d.CancelRequested == 1
        app.CancelFlag = true;
        return
    end
    OCT_vol = OCT_vol(depthROI(1):depthROI(2),:,:);
    OCT_vol_reg = OCT_vol;

    % for fm = 1:numBscans
    %     imshow(mat2gray(OCT_vol(:,:,fm)));pause(0.1)
    % end

    d.Message = [msg, strcat("OCTA volume: ",OCTA_vol_fn)];
    msg = [msg, strcat("OCTA volume: ",OCTA_vol_fn)];
    outPut(app, ["OCTA volume: ", OCTA_vol_fn]);

    OCTA_vol = double(read_OCT_vol(app, fullfile(OCT_vol_fns(ind).folder, OCTA_vol_fn)));
    if d.CancelRequested == 1
        app.CancelFlag = true;
        return
    end
    OCTA_vol = OCTA_vol(depthROI(1):depthROI(2),:,:);
    OCTA_vol_reg = OCTA_vol;

    tic
    for layer = 1:numPoints
        if d.CancelRequested == 1
            app.CancelFlag = true;
            return
        end
        % add (mat2gray)???
        OCT_vol_reg(layer,:,:) = imwarp(squeeze(OCT_vol(layer,:,:)), tformTotal, 'OutputView', OutputView); 
        OCTA_vol_reg(layer,:,:) = imwarp(squeeze(OCTA_vol(layer,:,:)), tformTotal, 'OutputView', OutputView);
    end

    clearvars OCT_vol OCTA_vol
    outPut(app, num2str(toc))
    % disp(num2str(toc))

    figure, imshowpair(ref_enface, retina_reg);

    % Non-rigid registration
    d.Message = [msg,"Non-rigid registration"];
    [D, ~] = imregdemons((mat2gray(retina_reg)), (mat2gray(ref_enface)),[1000 600 300],'AccumulatedFieldSmoothing', 3);
    retina_reg = imwarp(retina_reg, D);

    figure, imshowpair(ref_enface, retina_reg);
    % imwrite(mat2gray(retina_reg), savefn_retina);

    for layer = 1:numPoints
        if d.CancelRequested == 1
            app.CancelFlag = true;
            return
        end
        OCT_vol_reg(layer,:,:) = imwarp(squeeze(OCT_vol_reg(layer,:,:)), D);
        OCTA_vol_reg(layer,:,:) = imwarp(squeeze(OCTA_vol_reg(layer,:,:)), D);
    end

    d.Message = [msg,"Axial matching..."];
    % axial matching
    numBatch = 20;
    [row_shift, col_shift] = matchAxial(ref_vol, OCT_vol_reg, numBatch, 1);

    for k = 1:numBscans
        for j = 1:numAscans
            if d.CancelRequested == 1
                app.CancelFlag = true;
                return
            end
            OCT_vol_axmat(:,j,k) = circshift(OCT_vol_reg(:,j,k),[row_shift(j,k),col_shift(j,k)]);
            OCTA_vol_axmat(:,j,k) = circshift(OCTA_vol_reg(:,j,k),[row_shift(j,k),col_shift(j,k)]);
        end
    end

    clearvars OCT_vol_reg OCTA_vol_reg

    OCT_vol_axmat_final = zeros(app.datasize(2),app.datasize(1),app.datasize(3));
    OCT_vol_axmat_final(depthROI(1):depthROI(2),:,:) = OCT_vol_axmat; clear OCT_vol_axmat

    OCTA_vol_axmat_final = zeros(app.datasize(2),app.datasize(1),app.datasize(3));
    OCTA_vol_axmat_final(depthROI(1):depthROI(2),:,:) = OCTA_vol_axmat; clear OCTA_vol_axmat

    retina_reg_Z(:,:,ind) = retina_reg;

    close all
    
    all_OCT = all_OCT + OCT_vol_axmat_final;
    all_OCTA = all_OCTA + OCTA_vol_axmat_final;

    d.Message = [msg,"Saving registered OCT/OCTA volume..."];
    % save OCT
    % saveAs3DTiffStack(OCT_vol_axmat_final, savefnTIF_OCT)
    saveAsIMG(savefnIMG_OCT, permute(flipud(OCT_vol_axmat_final),[2,1,3]));

    % save OCTA
    % saveAs3DTiffStack(OCTA_vol_axmat_final, savefnTIF_OCTA)
    saveAsIMG(savefnIMG_OCTA, permute(flipud(OCTA_vol_axmat_final),[2,1,3]));

    %save en face OCT (to check orientation)
    imwrite(mat2gray(squeeze(mean(OCT_vol_axmat_final))),savefn_projOCT_V);

    close all

    clearvars OCT_vol_axmat_final OCTA_vol_axmat_final
end

d.Message = strcat("Saving averaged data at timepoint ", app.tp);
save(fullfile(app.save_mat,strcat('registered_',app.tp,'_enfaceOCTA_fromZeiss.mat')),"retina_reg_Z",'-v7.3');
saveAs3DTiffStack(retina_reg_Z,fullfile(save_reg_proj,strcat('registered_',app.tp,'_enfaceOCTA_fromZeiss.tif')));

all_OCT= all_OCT./length(retina_fns);
saveAsIMG(fullfile(app.save_segmentation,strcat(app.subject,'_',app.tp,'_Angiography_3x3_cube_z_all.img')), permute(flipud(all_OCT),[2,1,3]));

all_OCTA = all_OCTA./length(retina_fns);
save(fullfile(app.save_mat,strcat(app.subject,'_',app.tp,'_Angiography_3x3_FlowCube_z_all.mat')),'all_OCTA','-v7.3');
saveAs3DTiffStack(all_OCTA, fullfile(app.save_averaged,strcat(app.subject,'_',app.tp,'_Angiography_3x3_FlowCube_z_all.tif')))

d.close;
outPut(app, "Registration completed");
% uiwait(msgbox('Successed.'));
end