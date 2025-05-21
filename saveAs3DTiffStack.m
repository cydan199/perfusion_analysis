function saveAs3DTiffStack(img_volume, output_filename)
    
    % Initialize the TIFF file
    t = Tiff(output_filename, 'w');
    
    % Set the TIFF tags
    tagstruct.ImageLength = size(img_volume, 1);
    tagstruct.ImageWidth = size(img_volume, 2);
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 8;
    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    
    % Write the slices to the TIFF file
    for idx = 1:size(img_volume, 3)
        if idx > 1
            writeDirectory(t);
        end
        setTag(t, tagstruct);
        write(t, uint8(squeeze(img_volume(:, :, idx))));
    end
    
    % Close the TIFF file
    close(t);
end