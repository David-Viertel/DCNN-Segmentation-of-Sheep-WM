image = data2;

z_axis = size(image, 3);

img = image;

% Assuming 'images' is your image dataset
for i = 1:z_axis
    % Read the image
   
    
 
    

    % Adjust the image intensity values to standardize them
    img(:,:,i) = imadjust(image(:,:,i));
    
    
end


figure, imshow(image(:,:,1)), title('original');

figure, imshow(img(:,:,1)), title('intensity adjused');