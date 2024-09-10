% Define your area threshold here
areaThreshold = 5000;

% Initialize the 3D matrix for the updated binary images
updatedBinaryImage3D = false(size(FO05_label));
new_image3D = zeros(size(FO05_standard));

for i = 1:size(FO05_binary, 3)
    % Step 1: Identify and remove large structures from the separate binary image
    labeledImage = bwlabel(FO05_binary(:,:,i));
    stats = regionprops(labeledImage, 'Area');
    areas = [stats.Area];

    % Create a mask for small structures
    smallStructuresMask = ismember(labeledImage, find(areas < areaThreshold));

    % Step 2: Add the small structures to the original binary segmentation mask
    updatedBinaryImage = FO05_label(:,:,i) | smallStructuresMask;
    updatedBinaryImage3D(:,:,i) = updatedBinaryImage;

    % Step 3: Update the original image to retain small structures
    new_image = FO05_standard(:,:,i);
    new_image(updatedBinaryImage == false) = 0;
    new_image3D(:,:,i) = new_image;
end

% Display the results for the first slice as an example
figure; imshow(new_image3D(:,:,1));
figure; imshow(updatedBinaryImage3D(:,:,1));


% % Optional: Scale the final binary image if needed
% final_binary = new_image3D * 100;

