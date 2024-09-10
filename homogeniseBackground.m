N = 14;  % Replace with your actual number of slices

image = FO05_denoised_inf;


% Initialize an empty cell array to store the filenames

figure, imshow(image(:,:,1)), title('Binary Image');

se = strel('disk', 60);
J = uint8(zeros(size(image)));
I_new = uint8(zeros(size(image)));
% Loop over each image in the stack
for k = 1:N
    % Perform image opening on the k-th image in the stack
    J(:, :, k) = imopen(image(:, :, k), se);

    % Subtract the opened image from the original image
    I_new(:,:,k) = image(:, :, k) - J(:, :, k);

end


figure, imshow(J(:,:,1)), title('Binary Image');

figure, imshow(I_new(:,:,1));