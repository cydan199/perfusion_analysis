function saveAsTiffStackAbs(volume,filename)

if exist(filename,"file")
    delete(filename);
end

for i = 1:size(volume, 3)

    image = imadjust(mat2gray((volume(:,:,i))));

    imwrite(image, filename,'WriteMode',"append");
end