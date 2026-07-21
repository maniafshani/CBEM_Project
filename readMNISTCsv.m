function data = readMNISTCsv(filename)
%READMNISTCSV Read and validate one Moodle MNIST CSV file.
%
% Every row must contain:
%   column 1:     digit label from 0 to 9
%   columns 2-785: 784 pixel intensities from 0 to 255

    if ~isfile(filename)
        error(['MNIST file not found:\n%s\n\nPlace the Moodle CSV files ' ...
               'inside the project data folder or pass a custom dataDir.'], ...
              filename);
    end

    fprintf('Reading %s\n', filename);
    data = readmatrix(filename);

    if isempty(data)
        error('The file is empty: %s', filename);
    end

    if size(data, 2) ~= 785
        error(['Every MNIST row must contain 785 values: one label and ' ...
               '784 pixels. File %s has %d columns.'], ...
              filename, size(data, 2));
    end

    if any(isnan(data(:)))
        error('The file contains missing or nonnumeric values: %s', filename);
    end

    labels = data(:, 1);
    pixels = data(:, 2:end);

    if any(labels < 0 | labels > 9 | labels ~= floor(labels))
        error('All labels must be integer digits from 0 to 9.');
    end

    if any(pixels(:) < 0 | pixels(:) > 255)
        error('All pixel intensities must be between 0 and 255.');
    end

    fprintf('  loaded %d rows and %d columns\n', ...
            size(data, 1), size(data, 2));
end
