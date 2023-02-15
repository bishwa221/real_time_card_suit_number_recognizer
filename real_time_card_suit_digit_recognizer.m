runApplication();

function runApplication()
    cam = webcam('EpocCam');
    preview(cam)
    
    showTimer=timer('timerFcn', @startProcessing);
    set(showTimer, 'ExecutionMode', 'fixedRate', 'Period', 15);    
    start(showTimer)

    
    function startProcessing(varargin)
        try
            img = snapshot(cam);
            processedImage = processImage(img);
            perdictCard(processedImage);
        catch er
            fprintf('Error: %s \n', er.message);
        end
    end
end

function cropped_image = processImage(img)
%     originalImage = imresize(img, 1);
    originalImage = img;
    gray = rgb2gray(originalImage);

    img_ad = imadjust(gray);
    img_filt = medfilt2(img_ad);
%     imshow(img_filt)

    bin_img = imbinarize(img_filt,0.7);
    
%     figure
%     imshow(bin_img)
    
    angleList = zeros(1, 100);
    for i = 1 : numel(angleList)
        [x1, y1, x2, y2] = getSlopePoints(bin_img);
        slope = getSlope(x1, y1, x2, y2);
        angleList(1,i) = getAngle(slope);
    end

    [angle, freq] = mode(angleList, 2);

    if(freq == 1)
        angle = mean(angleList, 2);
    end
    
    rotated_image = imrotate(bin_img, angle);
%     imshow(rotated_image)
    
    % Get corners
    y1 = 0;
    y2 = 0;
    x1 = 0;
    x2 = 0;
    y_sum = sum(rotated_image, 2); %adds along rows, returns one column
    y_len = numel(y_sum);
    threshold = 30;
    
    for y = 1:y_len
        if(y_sum(y)> threshold)
            y1 = y;
            break
        end
    end
    
    for y = 1:numel(y_sum)
        if(y_sum(y_len - y - 1) > threshold)
            y2 = y_len - y - 1;
            break
        end
    end
    
    x_sum = sum(rotated_image); %adds in a column, return rows
    x_len = numel(x_sum);
    for x = 1:x_len
        if(x_sum(x) > threshold)
            x1 = x;
            break
        end
    end
    
    for x = 1:x_len
        if(x_sum(x_len - x - 1) > threshold)
            x2 = x_len - x - 1;
            break
        end
    end
    
    rotated_image = imrotate(originalImage, angle);
    cropped_image = imcrop(rotated_image, [x1 y1 x2-x1 y2-y1]);
    if (y2-y1 < x2-x1)
        cropped_image = imrotate(cropped_image,-90);
    end

    cropped_image = imresize(cropped_image,[640, 420]);
    cropped_image = imcrop(cropped_image, [10 20 399 599]);

    figure
    imshow(cropped_image)
    imwrite(cropped_image, 'test.jpg')
end

function perdictCard(img)
    img = imresize(img, [600, 1000]);
    spades = imbinarize(imread("temp/spades.jpg"),0.5);
    clubs = imbinarize(imread("temp/clubs.jpg"),0.5);
    hearts = imbinarize(imread("temp/hearts.jpg"),0.5);
    diamonds = imbinarize(imread("temp/diamonds.jpg"),0.5);

    suits_templates = [spades, clubs, hearts, diamonds];

    ace = imbinarize(imread("temp/ace.jpg"),0.5);
    two = imbinarize(imread("temp/two.jpg"),0.5);
    three = imbinarize(imread("temp/three.jpg"),0.5);
    four = imbinarize(imread("temp/four.jpg"),0.5);
    five = imbinarize(imread("temp/five.jpg"),0.5);
    six = imbinarize(imread("temp/six.jpg"),0.5);
    seven = imbinarize(imread("temp/seven.jpg"),0.5);
    eight = imbinarize(imread("temp/eight.jpg"),0.5);
    nine = imbinarize(imread("temp/nine.jpg"),0.5);
    ten = imbinarize(imread("temp/ten.jpg"),0.5);
    jack = imbinarize(imread("temp/jack.jpg"),0.5);
    queen = imbinarize(imread("temp/queen.jpg"),0.5);
    king = imbinarize(imread("temp/king.jpg"),0.5);
    
    num_templates = [ace,two, three,four,five,six,seven,eight,nine,ten,jack,queen,king];
    %% Image Thresholding
    
    img_gray = rgb2gray(img);
    img_ad = imadjust(img_gray);
    img_filt = medfilt2(img_ad);

    img_bin = imbinarize(img_filt);
%     imshow(img_bin)
% se = strel('rectangle', [30 30]);
%     img_bin = imerode(img_bin, se);

    img_2 = 1 -img_bin;
%     se = strel('rectangle', [8 8]);
%     eroded = imerode(img_2, se);
%     img_2 = eroded;
%     figure
%     imshow(img_2)
% img_bin = img;
% img_2 = img;
    %% Extract croping row and col
    try
    [~, col_bin] = size(img_2);
    sum_bin = sum(img_2);
    a = 1;
    while (a <= col_bin)
        if (sum_bin(a) == 0) %stops after first black column found
            crop_start_col = a;
            break
        end
        a = a+1;
    end
    %%
    
    b = crop_start_col + 1;
    while (sum_bin(b)==0) %stops after first white column encountered
        b = b+1;
    end
    %%
    while (b<=col_bin)
        if (sum_bin(b)==0)
            crop_end_col = b; %stops after second black column encountered
            break
        end
        b = b+1;
    end
    %% Extract number and suit
    img_temp = img_bin(1:200,crop_start_col:crop_end_col);
    %imshow(img_temp)
    [temp_row, temp_col] = size(img_temp);
%     img_filt_2 = medfilt2(img_temp);
%     imshow(img_filt_2);
    img_temp = 1 - img_temp;
%     figure
%     imshow(img_temp)
    %% get first row and last row for number and suit
    
    sum_ver = sum(img_temp);
    sum_hor = sum(img_temp,2);
    %%
    
    % extract first row
    i = 2;
    first_num_row = 5;
    while (i <= temp_row)
        if (sum_hor(i)~=0)
            first_num_row = i;
            break
        end
        i = i+1;
    end
    
    %extract last row
    j = first_num_row + 1;
    last_num_row = 80;
    while (j<=temp_row)
        if (sum_hor(j) == 0)
            last_num_row = j-1;
            break
        end
        j= j+1;
    end
    
    %extract first suit row
    k = last_num_row + 1;
    first_suit_row = 85;
    while (k<=temp_row)
        if (sum_hor(k) ~= 0)
            first_suit_row = k;
            break
        end
        k = k + 1;
    end
    
    % extract last suit row
    last_suit_row = 145;
    m = first_suit_row + 1;
    while (m<=temp_row)
        if (sum_hor(m) == 0)
            last_suit_row = m-1;
            break
        end
        m= m+1;
    end    
        
    %% get first and last columns
    
    % extract first col
    n = 2;
    first_num_col = 25;
    while (n <= temp_col)
        if (sum_ver(n)~=0)
            first_num_col = n;
            break
        end
        n = n+1;
    end
    
    %extract last col
    last_num_col = 125;
    p = first_num_col + 1;
    while (p<=temp_col)
        if (sum_ver(p) == 0)
            last_num_col = p;
            break
        end
        p= p+1;
    end

    catch
        first_num_row = 5;
        last_num_row = 80;
        first_num_col = 25;
        last_num_col = 125;
        first_suit_row = 85;
        last_suit_row = 145;
    end


    %%
    num_test = img_temp(first_num_row:last_num_row, first_num_col:last_num_col);
    suit_test = img_temp(first_suit_row:last_suit_row,first_num_col:last_num_col);
    num_test = imresize(num_test,[100,100]);
    suit_test = imresize(suit_test,[100,100]);
    num_test = imgaussfilt(num_test);
    suit_test = imgaussfilt(suit_test);
    
    figure
    imshow(num_test)
    figure
    imshow(suit_test)

    %% Template matching for all suits
    
    suit_cc = zeros(1,4);
    % s c h d
    for i = 1:4
        start_col = ((i-1)*100) + 1;
        end_col = ((i-1)*100) + 100;
        suit_temp = suits_templates(:, start_col:end_col);
        cc = normxcorr2(suit_temp,suit_test);
        [max_cc,~] = max(abs(cc(:)));
        suit_cc(1,i) = max_cc;
    end
    [~,I] = max(suit_cc);
    suits_names = ["spades", "clubs", "hearts", "diamonds"];
    fprintf("Suit of the card is: %s \n", suits_names(I))
    %% Template matching for all numbers
    
    num_cc = zeros(1,13);
    % ace two three four
    for i = 1:13
        start_col = ((i-1)*100) + 1;
        end_col = ((i-1)*100) + 100;
        num_temp = num_templates(:, start_col:end_col);
        cc = normxcorr2(num_temp,num_test);
        [max_cc,~] = max(abs(cc(:)));
        num_cc(1,i) = max_cc;
    end
    [~,I_num] = max(num_cc);
    num_names = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"];
    fprintf("Number of the card is: %s \n", num_names(I_num))
end

function angle = getAngle(slope)
    % theta = atan((s2-s1)/(1+(s2*s1)));
    % Since s2 is always 0 for our case
    % we use theta = atan(slope)
    theta = atan(slope);
    angle = rad2deg(theta);
end

function m = getSlope(x1,y1,x2,y2)
    m = (y2-y1)/(x2-x1);
end

function [x1,y1,x2,y2] = getSlopePoints(img)
    x1 = 0; %col_1
    x2 = 0;
    y1 = 0; %row_1
    y2 = 0;
    [row,col] = size(img);

    for y = 1:row
        if (sum(img(y,:)) > 2) && (sum(img(y+40,:)) > 2) && (sum(img(y+60,:)) > 2)
            y1 = y + 40;
            y2 = y + 60;
            i = 1;
            j = 1;
            while (i<=col)
                el = img(y1,i);
                if (el~=0)
                    x1 = i;
                    break
                end
                i = i+1;
            end
            
            while (j<=col)
                el = img(y2,j);
                if (el~=0)
                    x2 = j;
                    break
                end
                j = j+1;
            end
            break
           
        end
    end
end