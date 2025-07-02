%**xulu������ع�
% ��ָ���ļ�����������ȡ��raw�����ݣ���ȡ������ɢ�����ı�����'03_Training_ASC'�ļ��У�
% ��������ɢ�������ؽ���Ӱ����Ƭ������'03_Training_ASC_reconstruct'�ļ���
% �ű����Զ���������Ŀ¼�µ��������ļ��У�������Ŀ¼�ṹ���䡣
clear;
clc;
close all;

% --- �û����� ---
% �����Ŀ¼�������л�������Ŀ��λ��
project_root = 'E:\Document\paper_library\3rd_paper_250512\code\ASCE_Net\datasets\SAR_ASC_Project\';

% ����Ŀ¼������.raw�ļ���Ŀ¼
input_dir = fullfile(project_root, '02_Data_Processed_raw');

% ���Ŀ¼
output_asc_dir = fullfile(project_root, '03_Training_ASC'); % ���ɢ������
output_recons_dir = fullfile(project_root, '03_Training_ASC_reconstruct'); % ����ؽ����
output_jpeg_dir = fullfile(project_root, '03_Training_ASC_reconstruct_jpeg'); % ���JPEG��ʽ���ؽ�ͼ

% --- ������ ---
% �������Ŀ¼�Ƿ����
if ~isfolder(input_dir)
    error('����Ŀ¼������: %s', input_dir);
end

% �ݹ��������.raw�ļ�
fprintf('���ڴ� %s �в���.raw�ļ�...\n', input_dir);
files = dir(fullfile(input_dir, '**', '*.raw'));
fprintf('�ҵ��� %d ��.raw�ļ���\n', length(files));

for i = 1:length(files)
    input_raw_file = fullfile(files(i).folder, files(i).name);
    fprintf('���ڴ����ļ� (%d/%d): %s\n', i, length(files), input_raw_file);
    
    % --- 1. �������·���������ļ��� ---
    % ��ȡ���·��
    relative_path = erase(files(i).folder, input_dir);
    
    % ����ɢ���������Ŀ¼
    current_output_asc_dir = fullfile(output_asc_dir, relative_path);
    if ~isfolder(current_output_asc_dir)
        mkdir(current_output_asc_dir);
    end
    
    % �����ؽ�������Ŀ¼
    current_output_recons_dir = fullfile(output_recons_dir, relative_path);
    if ~isfolder(current_output_recons_dir)
        mkdir(current_output_recons_dir);
    end
    
    % ����JPEG�ؽ�������Ŀ¼
    current_output_jpeg_dir = fullfile(output_jpeg_dir, relative_path);
    if ~isfolder(current_output_jpeg_dir)
        mkdir(current_output_jpeg_dir);
    end
    
    % ��ȡ������չ�����ļ���
    [~, basename, ~] = fileparts(files(i).name);

    % --- 2. ��ȡ�ʹ���ͼ�� ---
    [fileimage, image_value] = image_read(input_raw_file);
    
    % �㷨1����������ȡɢ������
    scatter_all = extrac(fileimage, image_value);
    
    % ������ȡ������ɢ������
    asc_output_filename = [basename, '_yang.mat'];
    save(fullfile(current_output_asc_dir, asc_output_filename), 'scatter_all');
        
    % --- 3. ����ؽ��ͱ��� ---
    s = simulation(scatter_all); % ͼ1���ؽ�ͼ
    diff = fileimage - s;
    
    % �����ؽ�����Ƭ�Ͳ��� (.mat��ʽ)
    recons_output_filename = [basename, '_yangRecon.mat'];
    save(fullfile(current_output_recons_dir, recons_output_filename), 's', 'diff');
    
    % ���ؽ�ͼ���һ��������ΪJPEG
    s_normalized = mat2gray(s); % ������s��һ����[0, 1]��Χ
    jpeg_output_filename = [basename, '_reconstruct.jpg'];
    imwrite(s_normalized, fullfile(current_output_jpeg_dir, jpeg_output_filename));
    
    % �ر����д򿪵�ͼ�񴰿ڣ����ⵯ����������
    close all;
end

fprintf('�����ļ�������ϡ�\n');

