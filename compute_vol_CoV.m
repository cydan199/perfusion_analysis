function compute_vol_CoV(app, depthROI, ftemp)
app.CancelFlag = false;

fig = app.UIFigure;
d = uiprogressdlg(fig,'Title','Computing B-scan-based CoV',...
    'Indeterminate','on','Cancelable','on');
drawnow

numPoints = depthROI(2)-depthROI(1)+1;
numBscans = app.datasize(3);
numAscans = app.datasize(1);


Bscans_CoV = zeros(numPoints,numAscans,numBscans);

% create all saving folders
% savefns = fullfile(app.save_folder, app.tp, 'mat');
% if ~exist(savefns,'dir')
%     mkdir(savefns)
% end

OCTA_vol_fns_not_ordered = dir(fullfile(app.save_registered,'*_FlowCube_z*.img'));
OCTA_vol_fns = natsortfiles(OCTA_vol_fns_not_ordered); clear OCTA_vol_fns_not_ordered

ref_frames = zeros(numPoints,numAscans,length(OCTA_vol_fns));

for fnum = 1:numBscans
    d.Message = strcat("Calculating CoV at frame: ", num2str(fnum));

    for vol = 1:length(OCTA_vol_fns)
        OCTA_vol = double(read_OCT_vol(app, fullfile(OCTA_vol_fns(vol).folder, OCTA_vol_fns(vol).name)));
        if d.CancelRequested == 1
            app.CancelFlag = true;
            return
        end
        OCTA_vol = OCTA_vol(depthROI(1):depthROI(2),:,:);

        ref_frames(:,:,vol) = OCTA_vol(:,:,fnum);
    end

    % save  of three volumes for check
    if fnum == ftemp
        if d.CancelRequested == 1
            app.CancelFlag = true;
            return
        end
        saveAs3DTiffStack(ref_frames,fullfile(app.save_folder,app.tp,'Cropped_V',strcat('frame_',num2str(ftemp),'_fromVolume.tif')));
    end

    for ii = 1:size(ref_frames,1)
        for jj = 1:size(ref_frames,2)
            if d.CancelRequested == 1
                app.CancelFlag = true;
                return
            end

            Bscans_CoV(ii,jj,fnum) = getCV(squeeze(ref_frames(ii,jj,:)));
        end
    end
end

d.Message = "Computation completed, saving results... ";
save(fullfile(app.save_mat,'CoV_fromBscans.mat'),'Bscans_CoV','-v7.3');

d.close;
outPut(app, "CoV compuation completed");
end

