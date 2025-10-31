import os
import json

def process_data_for_model(input_filename='line7_dir1.json', output_filename='normalized_model_data.json'):
    """
    读取JSON文件，提取tx_temp, rx_temp，并对diff_before_old_weight矩阵进行归一化处理，
    最终将所有结果保存到一个新的JSON文件中。

    Args:
        input_filename (str): 输入的JSON文件名。
        output_filename (str): 输出的JSON文件名。
    """
    try:
        # 1. 读取并解析输入的JSON文件
        with open(input_filename, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(f"成功读取文件 '{input_filename}'。")

        output_data = {}

        # 2. 遍历JSON文件中的每一条记录
        for key, value in input_data.items():
            if isinstance(value, dict):
                tx_temp = value.get('tx_temp')
                rx_temp = value.get('rx_temp')
                diff_matrix = value.get('diff_before_old_weight')

                if all(v is not None for v in [tx_temp, rx_temp, diff_matrix]):
                    
                    # 3. 计算矩阵元素的总和 (sum_val)
                    sum_val = sum(sum(row) for row in diff_matrix)

                    normalized_matrix = []
                    # 4. 进行归一化处理
                    if sum_val == 0:
                        # 如果总和为0，则归一化矩阵也为全0，直接复制即可
                        normalized_matrix = [row[:] for row in diff_matrix]
                    else:
                        # 矩阵中的每个元素都除以sum_val
                        for row in diff_matrix:
                            normalized_row = [element / sum_val for element in row]
                            normalized_matrix.append(normalized_row)
                    
                    # 5. 按照指定格式构建输出字典
                    output_data[key] = {
                        'tx_temp': tx_temp,
                        'rx_temp': rx_temp,
                        'normalized_matrix': normalized_matrix,
                        'sum_val': sum_val
                    }

        # 6. 将处理后的数据写入新的JSON文件
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        print(f"数据处理完成，保存到 '{output_filename}' 文件中。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_filename}'。请确保文件与脚本在同一目录下。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_filename}' 不是有效的JSON格式。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# # 运行处理函数
# process_data_for_model()

mering_folder = "/work/hwc/SENSOR_STAGE2/training_data/test_line/merging"
nonmering_folder = "/work/hwc/SENSOR_STAGE2/training_data/test_line/nonmerging"
save_fold = "/work/hwc/SENSOR_STAGE2/training_data/test_line/norm_data"
if os.path.exists(save_fold):
    pass
else:
    os.makedirs(save_fold)

filenames = os.listdir(mering_folder)  # 文件名相同

for filename in filenames:
    mering_file_path = os.path.join(mering_folder, filename)
    nonmering_file_path = os.path.join(nonmering_folder, filename)
    prefix = filename.split(".")[0]
    merging_file_name = prefix + ".json"
    norm_merging_file_path = os.path.join(save_fold, merging_file_name)
    nonmering_file_name = "nonmerging_" + prefix + ".json"
    norm_nonmering_file_path = os.path.join(save_fold, nonmering_file_name)
    process_data_for_model(mering_file_path, norm_merging_file_path)
    # process_data_for_model(nonmering_file_path, norm_nonmering_file_path)


