function out = getFeatureMap(im_patch, feature_type, cf_response_size, hog_cell_size,w2c)

% code from DSST

% allocate space
switch feature_type
    case 'fhog'
        temp = fhog(single(im_patch), hog_cell_size);
        h = cf_response_size(1);
        w = cf_response_size(2);
        out = zeros(h, w, 38, 'single');
        out(:,:,2:28) = temp(:,:,1:27);
        if hog_cell_size > 1
            im_patch_rsz = mexResize(im_patch, [h, w] ,'auto');
        end
        % if color image
        if size(im_patch_rsz, 3) > 1
            im_patch_gray = rgb2gray(im_patch_rsz);
            out(:,:,1) = single(im_patch_gray)/255 - 0.5;
            CN = im2c(single(im_patch_rsz), w2c, -2);
        % put colornames features into output structure
            out(:,:,29:38) = CN;
        else
            out(:,:,1) = single(im_patch_rsz)/255 - 0.5;
        end
        
    case 'gray'
        if hog_cell_size > 1, im_patch = mexResize(im_patch,cf_response_size,'auto');   end
        if size(im_patch, 3) == 1
            out = single(im_patch)/255 - 0.5;
        else
            out = single(rgb2gray(im_patch))/255 - 0.5;
        end        
end
        
end

