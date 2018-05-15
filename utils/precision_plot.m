function [precisions,success] = precision_plot(positions, ground_truth, title, show)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	
	max_threshold = 50;  %used for graphs in the paper
	
	
	precisions = zeros(max_threshold, 1);
	
	if size(positions,1) ~= size(ground_truth,1),
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
		%just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames
	distances = sqrt((positions(:,1) - ground_truth(:,1)+positions(:,3) - ground_truth(:,3)).^2 + ...
				 	 (positions(:,2) - ground_truth(:,2)+positions(:,4) - ground_truth(:,4)).^2);
	distances(isnan(distances)) = [];

    z1 = max([positions(:,1),ground_truth(:,1)],[],2);
    z2 = min([positions(:,1)+positions(:,3),ground_truth(:,1)+ground_truth(:,3)],[],2);
    z3 = max([positions(:,2),ground_truth(:,2)],[],2);
    z4 = min([positions(:,2)+positions(:,4),ground_truth(:,2)+ground_truth(:,4)],[],2);
    s = (z2-z1).*(z4-z3);
    s(z2<z1) = 0;
    s(z4<z3) = 0;
    s1 = positions(:,3).*positions(:,4);
    s2 = ground_truth(:,3).*ground_truth(:,4);
    area = s./(s1+s2-s);
    
    for p = 1:100,
		success(p) = nnz(area >= p/100) / numel(area);
    end
    
	%compute precisions
	for p = 1:max_threshold,
		precisions(p) = nnz(distances <= p) / numel(distances);
	end
	
	%plot the precisions
	if show == 1,
		figure('Name',['Precisions - ' title])
		plot(precisions, 'LineWidth',2)
		xlabel('Threshold'), ylabel('Precision')
        figure('Name',['Success Rate - ' title])
		plot(success, 'LineWidth',2)
		xlabel('Threshold'), ylabel('Success')
	end
	
end

