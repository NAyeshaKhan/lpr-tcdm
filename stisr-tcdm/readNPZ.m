function data = readNPZ(filename)
    % Uses Python's numpy to load .npz files
    np = py.importlib.import_module('numpy');
    npz = np.load(filename);

    keys = cell(npz.keys().tolist());
    data = struct();

    for i = 1:length(keys)
        key = keys{i};
        arr = npz{key};
        data.(key) = double(arr); % converts to MATLAB numeric array
    end
end
