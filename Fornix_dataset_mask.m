for i = 1:29
    % Read the STL file
    filename = sprintf('05_FO_02_%d %d_001.stl', i, i);
    fv = stlread(filename);
    
    % Extract vertices
    vertices = fv.Points;

    % Define the output .txt filename
    txt_filename = sprintf('coordinates_%d_%d.txt', i, i);
    
    % Open a file for writing
    fileID = fopen(txt_filename, 'w');

    % Write the coordinates to the file
    fprintf(fileID, '%.6f %.6f %.6f\n', vertices');

    
    fclose(fileID);

    disp(['Coordinates have been written to ', txt_filename]);
end


% List of file names
files = dir('*.txt');

% Structuring element for dilation
se = strel('disk', 1);

%z axis
Z = 68;

rowsToDelete = 0;

% Loop over each file
for i = 1:length(files)
    % Read the file
    data = readmatrix(files(i).name);
    
    % Round z coordinate to the nearest 0.01 (only for stl to point cloud
    % files)
    % data(:,3) = round(data(:,3) / 0.01) * 0.01;

    % Find rows where the z coordinate is not divisible by 0.15
    rowsToDelete = mod(data(:,3), 0.15) ~= 0;

    % Delete those rows
    data(rowsToDelete, :) = [];

    data(:,1) = data(:,1)-0.01;
    data(:,2) = data(:,2)-0.01;

    % Round x and y coordinates to the nearest 0.02
    data(:,1) = round(data(:,1) / 0.02) * 0.02;
    data(:,2) = round(data(:,2) / 0.02) * 0.02;

    data(:,1) = data(:,1)/0.02;
    data(:,2) = data(:,2)/0.02;
    data(:,3) = data(:,3)/0.15;

    data = int32(data);

    data(:,1) = data(:,1) + 1;
    data(:,2) = data(:,2) + 1;
    data(:,3) = data(:,3) + 1;


    % Initialize a 3D matrix of zeros
    label = zeros(768, 1024, Z);

   
    % Set the corresponding elements in matrix3D to 1
    for j = 1:size(data, 1)
        label(data(j,2), data(j,1), data(j,3)) = 1;
    end


    % Dilate the image

    label_dilated = label;
    for m = 1:Z
        label_dilated(:,:,m) = imdilate(label(:,:,m), se);
    end

    % Fill the holes

    label_filled = label;
    for n = 1:Z
        label_filled(:,:,n) = imfill(label_dilated(:,:,n), 'holes');
    end
    
    % Get the file name without the extension
    [~, name] = fileparts(files(i).name);

    % Save the result
    save(['label_' name '.mat'], 'label_filled');
end



% Get a list of all .mat files in the current directory
matFiles = dir('*.mat');

% Initialize a matrix with zeros. The size should match your image size.
% For example, if your images are 256x256, you would do:
combinedData = zeros(size(FO05_raw));

% Loop through each .mat file
for i = 1:length(matFiles)
    % Load the .mat file
    data = load(matFiles(i).name);
    
    % Add the loaded data to the combinedData matrix
    % Assuming the data in each .mat file is stored in a variable named 'label'
    combinedData = combinedData + data.label_filled;
end


% Assume 'combinedData' is your matrix
% Find the indices where the matrix values are greater than 1
indices = combinedData > 1;

% Set those values to 1
combinedData(indices) = 1;

FO05_label = combinedData;


figure, imshow(FO05_label(:,:,1));

row = 742:768; % Rows
col = 976:1024; % Columns

row2 = 742:768; % Rows
col2 = 1:48; % Columns

row3 = 695:768; % Rows
col3 = 250:300; % Columns

row4 = 667:768; % Rows
col4 = 235:315; % Columns

row5 = 725:768; % Rows
col5 = 272:310; % Columns


% Set the pixel values in the corner region to black
FO05_label(row, col, :) = 0;
FO05_label(row2, col2, :) = 0;

FO05_label(row3, col3, 25:39) = 0;

FO05_label(row4, col4, 40:68) = 0;

FO05_label(row5, col5, 10:24) = 0;


% Display the image
figure, imshow(FO05_label(:,:,1));




% Save the combinedData matrix to a new .mat file
save('FO05_label.mat', 'FO05_label');
