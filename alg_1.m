clc;

load alg1.mat

cam = ipcam('http://192.168.43.1:8080/video');
% cam = webcam;
preview(cam) 
closePreview(cam)
disp('Ready')
pause(4)
x1 = snapshot(cam);
figure;
imshow(x1);

% [s a] = uigetfile('d');
% inp = strcat([a s]);
% x1 = imread(inp);
ds= augmentedImageDatastore(imageSize, x1, 'ColorPreprocessing', 'gray2rgb');
imagefeature = activations(net, ds, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

op = predict(classifier,imagefeature, 'ObservationsIn', 'columns');
out = sprintf('%s',op);

msgbox(out);