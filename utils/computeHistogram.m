function histogram = computeHistogram(patch, mask, n_bins, grayscale_sequence)

	[h, w, d] = size(patch);

	assert(all([h w]==size(mask)) == 1, 'mask and image are not the same size');

	bin_width = 256/n_bins;

	% convert image to 1d array with same n channels of img patch
	patch_array = reshape(double(patch), w*h, d);
	% compute to which bin each pixel (for all 3 channels) belongs to
	bin_indices = floor(patch_array/bin_width) + 1;

	if grayscale_sequence
		histogram = accumarray(bin_indices, mask(:), [n_bins 1])/sum(mask(:));
	else
		% the histogram is a cube of side n_bins
		histogram = accumarray(bin_indices, mask(:), [n_bins n_bins n_bins])/sum(mask(:));
	end

end
