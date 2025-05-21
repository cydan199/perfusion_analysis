function saveAsIMG(filename, data)
    fid = fopen(filename,'w');
    fwrite(fid, data, 'uint8');
    fclose(fid);
end