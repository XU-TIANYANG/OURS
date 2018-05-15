function [results] = trackerMain(p, im, bg_area, fg_area, area_resize_factor)
    num_frames = numel(p.img_files);
    OTB_rect_positions = zeros(num_frames, 4);
	pos = p.init_pos;
    target_sz = p.target_sz;
	agent_hf_den = cell(1,1,p.num_agent);
    agent_hf_num = cell(1,1,p.num_agent);
    w2c = load('w2crs.mat');
    w2c = w2c.w2crs; 
    patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    new_pwp_model = true;
    [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
    new_pwp_model = false;
    hann_window = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');
    output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
    y = gaussianResponse(p.cf_response_size, output_sigma);
    yf = fft2(y);
    %% SCALE ADAPTATION INITIALIZATION
    if p.scale_adaptation
        scale_factor = 1;
        base_target_sz = target_sz;
        scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
        ss = (1:p.num_scales) - ceil(p.num_scales/2);
        ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        ysf = single(fft(ys));
        if mod(p.num_scales,2) == 0
            scale_window = single(hann(p.num_scales+1));
            scale_window = scale_window(2:end);
        else
            scale_window = single(hann(p.num_scales));
        end;
        ss = 1:p.num_scales;
        scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
        if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
            p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
        end
        scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
        min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
        max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));
    end

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
    weight_mx = zeros(3,num_frames);
    weight_agent = zeros(p.num_agent,num_frames);
    t_imread = 0;
    t_visual = 0;
    tic;
    for frame = 1:num_frames
        if frame>1
%          if frame==315;
%              test=0;
%          end
            tic_imread = tic;
            im = imread([p.img_files{frame}]);
            t_imread = t_imread + toc(tic_imread);
            % extract patch of size bg_area and resize to norm_bg_area
            im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
            pwp_search_area = round(p.norm_pwp_search_area / area_resize_factor);
            % extract patch of size pwp_search_area and resize to norm_pwp_search_area
            im_patch_pwp = getSubwindow(im, pos, p.norm_pwp_search_area, pwp_search_area);
            % compute feature map
            xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size,w2c);
            xtf = fft2(xt);
            ss_num = conj(xtf_old) .* xtf / prod(p.cf_response_size);
            response_ss = real(ifft2(bsxfun(@rdivide, sum(ss_num,3), sum(new_hf_den, 3)+100*p.lambda)));
            % apply Hann window
            xt_windowed = bsxfun(@times, hann_window, xt);
            % compute FFT
            xtf = fft2(xt_windowed);
            hf = cellfun(@(hf_den_c,hf_num_c) bsxfun(@rdivide, hf_num_c, sum(hf_den_c, 3)+p.lambda)...
                , agent_hf_den,agent_hf_num, 'uniformoutput', false);
            
            response_cf = cellfun(@(hf_c) ensure_real(ifft2(sum(conj(hf_c) .* xtf, 3))),hf,'uniformoutput', false);

            % Crop square search region (in feature pixels).
            response_cf = cellfun(@(response_cf_c) cropFilterResponse(response_cf_c, ...
                floor_odd(p.norm_delta_area / p.hog_cell_size)),response_cf,'uniformoutput', false);
            response_ss = cropFilterResponse(response_ss, ...
                floor_odd(p.norm_delta_area / p.hog_cell_size));
            if p.hog_cell_size > 1
                % Scale up to match center likelihood resolution.
                response_cf = cellfun(@(response_cf_c) mexResize(response_cf_c, p.norm_delta_area,'auto')...
                    ,response_cf,'uniformoutput', false);
                response_ss = mexResize(response_ss, p.norm_delta_area,'auto');
            end

            [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
            likelihood_map(isnan(likelihood_map)) = 0;
            response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

            %% ESTIMATION
            response =  cellfun(@(response_cf_c) mergeResponses(response_cf_c, response_pwp, response_ss, p.merge_factor, p.merge_method),...
            response_cf,'uniformoutput', false);
            response = cell2mat(response); 
            response = sum(response,3);
            [row, col] = find(response == max(response(:)), 1);
            response_cf_est = sum(cell2mat(response_cf),3)/p.num_agent;
            weight_mx(:,frame)=[response_cf_est(row, col),response_pwp(row, col),response_ss(row, col)]';
            weight_cf_agent = cellfun(@(response_cf_c) max(response_cf_c(:)) ,response_cf,'uniformoutput', false);
            weight_cf_agent_temp =  cell2mat(weight_cf_agent);
            weight_agent(:,frame) = weight_cf_agent_temp(1,1,:)/sum(weight_cf_agent_temp(:));
            v_neighbors = response(mod(row + [-1, 0, 1] - 1, size(response,1)) + 1, col);
            h_neighbors = response(row, mod(col + [-1, 0, 1] - 1, size(response,2)) + 1);
            row = row + subpixel_peak(v_neighbors);
            col = col + subpixel_peak(h_neighbors);
            center = (1+p.norm_delta_area) / 2;
            pos = pos + ([row, col] - center) / area_resize_factor;
            rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

            %% SCALE SPACE SEARCH
            if p.scale_adaptation
                im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
                xsf = fft(im_patch_scale,[],2);
                scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
                recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
                %set the scale
                scale_factor = scale_factor * scale_factors(recovered_scale);
                if scale_factor < min_scale_factor
                    scale_factor = min_scale_factor;
                elseif scale_factor > max_scale_factor
                    scale_factor = max_scale_factor;
                end
                % use new scale to update bboxes for target, filter, bg and fg models
                target_sz = round(base_target_sz * scale_factor);
                avg_dim = sum(target_sz)/2;
                bg_area = round(target_sz + avg_dim);
                if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
                if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end

                bg_area = bg_area - mod(bg_area - target_sz, 2);
                fg_area = round(target_sz - avg_dim * p.inner_padding);
                fg_area = fg_area + mod(bg_area - fg_area, 2);
                area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
            end
        end
        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        % compute feature map, of cf_response_size
        xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size,w2c);
        % apply Hann window
        xt = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt);
        xtf_old = xtf;
		new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
		new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);
        if frame == 1
		    hf_den = new_hf_den;
		    hf_num = new_hf_num;
            for nn = 1 : p.num_agent
                agent_hf_den{1,1,nn} = hf_den;
                agent_hf_num{1,1,nn} = hf_num;
            end
        else
        	hf_den = (1 - p.learning_rate_cf) * hf_den + p.learning_rate_cf * new_hf_den;
	   	 	hf_num = (1 - p.learning_rate_cf) * hf_num + p.learning_rate_cf * new_hf_num;
            agent_rand=permute(mat2cell(randsample([0 1],p.num_agent,true,[0.7 0.3]),1,ones(1,p.num_agent)),[3,1,2]);
            agent_hf_den = cellfun(@(hf_den_c,agent_rand_c) (1 - p.learning_rate_cf*agent_rand_c)...
                * hf_den_c + p.learning_rate_cf *agent_rand_c* new_hf_den, agent_hf_den, agent_rand, 'uniformoutput', false);
            agent_hf_num = cellfun(@(hf_num_c,agent_rand_c) (1 - p.learning_rate_cf*agent_rand_c)...
                * hf_num_c + p.learning_rate_cf *agent_rand_c* new_hf_num, agent_hf_num, agent_rand, 'uniformoutput', false);
%             if frame ==150
%                 test = 0;
%             end
            [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
        end

        if p.scale_adaptation
            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            new_sf_num = bsxfun(@times, ysf, conj(xsf));
            new_sf_den = sum(xsf .* conj(xsf), 1);
            if frame == 1,
                sf_den = new_sf_den;
                sf_num = new_sf_num;
            else
                sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
                sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
            end
        end

        if frame==1, rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; end

        rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];

        OTB_rect_positions(frame,:) = rect_position;

        if p.fout > 0,  fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', rect_position(1),rect_position(2),rect_position(3),rect_position(4));   end

        %% VISUALIZATION
        tic_visual = tic;
    if p.visualization
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1,  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',3);
            rectangle('Position',rect_position_padded, 'EdgeColor','r', 'LineWidth',3);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',3);
            rectangle('Position',rect_position_padded, 'EdgeColor','r', 'LineWidth',3);%padding
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
        end
        drawnow
    end
    t_visual = t_visual + toc(tic_visual);
    end
    elapsed_time = toc;
    % save data for OTB-13 benchmark
    results.type = 'rect';
%     results.res = OTB_rect_positions;
    results.res = OTB_rect_positions(6:354,:);
    results.fps = num_frames/(elapsed_time - t_imread-t_visual);
    results.weight_mx = weight_mx;
    results.weight_agent = weight_agent;
    
    %
%     results.len =  num_frames;
%     results.annoBegin=p.startframe;
%     results.startFrame=p.startframe;
     results.len =  349;
    results.annoBegin=6;
    results.startFrame=6;

    
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end

function delta = subpixel_peak(p)
	%parabola model (2nd order fit)
	delta = 0.5 * (p(3) - p(1)) / (2 * p(2) - p(3) - p(1));
	if ~isfinite(delta), delta = 0; end
end  % endfunction
