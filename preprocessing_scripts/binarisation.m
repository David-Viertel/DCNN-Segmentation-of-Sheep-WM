image = FO05_homo_inf;

z_axis = size(image, 3);

BW = uint8(zeros(size(image)));
T = adaptthresh(image, 0.5);
for k = 1:z_axis
    
    % Calculate adaptive threshold for the k-th image in the stack
    
   
    % Convert the k-th image to binary using the threshold
    BW(:, :, k) = imbinarize(image(:,:,k), T(:,:,k))*255;
end


BW2 = uint8(zeros(size(image)));

for k = 1:z_axis

    BW2(:,:,k) = bwareaopen(BW(:,:,k), 1000)*255;
end

figure, imshow(BW(:,:,1)), title('Binary Image');


figure, imshow(BW2(:,:,1)), title('Binary Image');
