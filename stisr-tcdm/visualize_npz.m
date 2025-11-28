function visualize_npz()
    % Load NPZ
    data_path = "diff_samples/hr_samples/1000_samples_1.npz";
    data = readNPZ(data_path);  % requires helper below

    % Display available keys and shapes
    keys = fieldnames(data);
    for i = 1:length(keys)
        key = keys{i};
        arr = data.(key);
        fprintf('%s: size = %s, class = %s\n', key, mat2str(size(arr)), class(arr));
    end

    % Images are stored in "arr_0" column
    images = data.arr_0;

    % Print statistics
    fprintf("Min: %.4f, Max: %.4f, Mean: %.4f, Std: %.4f\n", ...
        min(images(:)), max(images(:)), mean(images(:)), std(double(images(:))));

    fprintf("Generated samples: %s\n", mat2str(size(images)));

    % Saving first 10 images
    save_dir = "saved_images";
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    num_to_save = min(10, size(images, 1));
    for i = 1:num_to_save
        img = squeeze(images(i, :, :, :));
        % Display + save
        imshow(img);
        axis off;

        save_path = fullfile(save_dir, sprintf("image_%d.png", i-1));
        exportgraphics(gca, save_path, 'Resolution', 150);
        fprintf("Saved: %s\n", save_path);
    end
end
