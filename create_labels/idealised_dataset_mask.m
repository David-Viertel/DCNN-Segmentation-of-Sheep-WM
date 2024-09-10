% Load the image
imgStack = 255 - img_3d(1:1024,1:1024,1:200);
imgStack_noise = imgStack;
imgStack_noisemore = imgStack;


% Define a wide range for the radius
minRadius = 7; % Smallest possible radius
maxRadius = round(min(size(img, 1), size(img, 2)) / 2); % Largest possible radius based on image size



% Initialize the 3D mask stack for non-deleted circles
maskStack = false(size(imgStack));

% Process each slice in the stack
for slice = 1:size(imgStack, 3)
    
    disp(slice)
    img = imgStack(:, :, slice);

    % Convert to grayscale if the image is RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % Detect circles
    [centers, radii] = imfindcircles(img, [minRadius maxRadius]);

    % Number of circles to delete
    numCircles = size(centers, 1);
    numToDelete = floor(4 * numCircles / 5);

    % Randomly select three-fourths of the circles to delete
    deleteIdx = randperm(numCircles, numToDelete);

    % Create a mask for the circles to delete
    deleteMask = false(size(img, 1), size(img, 2));
    for i = 1:numToDelete
        deleteMask = deleteMask | createCirclesMask(size(img), centers(deleteIdx(i), :), radii(deleteIdx(i)));
    end

    % Create a mask for the remaining circles
    remainingMask = false(size(img, 1), size(img, 2));
    remainingCenters = centers(~ismember(1:numCircles, deleteIdx), :);
    remainingRadii = radii(~ismember(1:numCircles, deleteIdx));
    for i = 1:length(remainingRadii)
        remainingMask = remainingMask | createCirclesMask(size(img), remainingCenters(i, :), remainingRadii(i));
    end

    % Store the remaining mask in the 3D mask stack
    maskStack(:, :, slice) = remainingMask;

    % Delete the selected circles
    img(deleteMask) = 0; 
    % Delete the stuff not detected in main mask
    img(~remainingMask) = 0;


    img_temp = bwareaopen(img,35);
    blackPixels = img_temp == 0;

    img(blackPixels) = 0;

    img_noise = img;
    img_noisemore = img;
    noise = randi([0, 40], size(img));
    noise_more = randi([0, 80], size(img));
    img_noise(blackPixels) = noise(blackPixels);
    img_noisemore(blackPixels) = noise_more(blackPixels);


    % Draw thicker borders for the remaining circles
    thickness = 14; % Adjust the thickness as needed
    for i = 1:length(remainingRadii)
        newRadius = remainingRadii(i) - thickness / 2; % Adjust radius to grow inwards
        img = insertShape(img, 'Circle', [remainingCenters(i, :) newRadius], 'LineWidth', thickness, 'Color', 'white');
        img_noise = insertShape(img_noise, 'Circle', [remainingCenters(i, :) newRadius], 'LineWidth', thickness, 'Color', 'white');
    end

    % Convert back to grayscale if the image was converted to RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    if size(img_noise, 3) == 3
        img_noise = rgb2gray(img_noise);
    end

    % Save the modified slice back to the stack
    imgStack(:, :, slice) = img;
    imgStack_noise(:,:,slice) = img_noise;
    imgStack_noisemore(:,:,slice) = img_noisemore;
end




% Helper function to create a mask for a circle
function mask = createCirclesMask(imgSize, center, radius)
    [X, Y] = ndgrid(1:imgSize(1), 1:imgSize(2));
    mask = (X - center(2)).^2 + (Y - center(1)).^2 <= radius^2;
end

