#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inference.h"     // 包含推理函数的头文件
#include "model_weights.h" // 包含模型权重的头文件 (由 python 生成)
#include "cJSON.h"         // 包含 cJSON 库的头文件

// 函数：从文件中读取内容到字符串
char* read_file_to_string(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buffer = (char *)malloc(length + 1);
    if (!buffer) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(f);
        return NULL;
    }
    fread(buffer, 1, length, f);
    buffer[length] = '\0';
    fclose(f);
    return buffer;
}

// 主函数
int main(int argc, char *argv[]) {
    // 检查命令行参数
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <json_file_path> <sample_id>\n", argv[0]);
        return 1;
    }
    const char* json_filepath = argv[1];
    const char* sample_id_str = argv[2];

    // 1. 读取 JSON 文件内容
    char* json_string = read_file_to_string(json_filepath);
    if (!json_string) {
        return 1;
    }

    // 2. 解析 JSON 字符串
    cJSON *root = cJSON_Parse(json_string);
    free(json_string); // 释放文件内容字符串
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        return 1;
    }

    // 3. 查找对应的样本 ID
    cJSON *sample = cJSON_GetObjectItemCaseSensitive(root, sample_id_str);
    if (!cJSON_IsObject(sample)) {
        fprintf(stderr, "Error: Sample ID '%s' not found or not an object in %s\n", sample_id_str, json_filepath);
        cJSON_Delete(root);
        return 1;
    }

    // 4. 提取 "merging" -> "normalized_matrix"
    cJSON *merging = cJSON_GetObjectItemCaseSensitive(sample, "merging");
    if (!cJSON_IsObject(merging)) {
        fprintf(stderr, "Error: 'merging' object not found in sample '%s'\n", sample_id_str);
        cJSON_Delete(root);
        return 1;
    }
    cJSON *matrix_json = cJSON_GetObjectItemCaseSensitive(merging, "normalized_matrix");
    if (!cJSON_IsArray(matrix_json)) {
        fprintf(stderr, "Error: 'normalized_matrix' not found or not an array in sample '%s'\n", sample_id_str);
        cJSON_Delete(root);
        return 1;
    }

    // 5. 将 JSON 矩阵转换为 C 的 float 数组
    //    (假设维度固定为 ROWS x COLS)
    float sensor_data[ROWS][COLS];
    int current_row = 0;
    cJSON *row_json;
    cJSON_ArrayForEach(row_json, matrix_json) {
        if (!cJSON_IsArray(row_json) || current_row >= ROWS) {
            fprintf(stderr, "Error: Invalid matrix structure (row %d)\n", current_row);
            cJSON_Delete(root);
            return 1;
        }
        int current_col = 0;
        cJSON *cell_json;
        cJSON_ArrayForEach(cell_json, row_json) {
            if (!cJSON_IsNumber(cell_json) || current_col >= COLS) {
                fprintf(stderr, "Error: Invalid matrix structure (row %d, col %d)\n", current_row, current_col);
                cJSON_Delete(root);
                return 1;
            }
            sensor_data[current_row][current_col] = (float)cJSON_GetNumberValue(cell_json);
            current_col++;
        }
        if (current_col != COLS) {
             fprintf(stderr, "Error: Incorrect number of columns in row %d (expected %d, got %d)\n", current_row, COLS, current_col);
             cJSON_Delete(root);
             return 1;
        }
        current_row++;
    }
     if (current_row != ROWS) {
        fprintf(stderr, "Error: Incorrect number of rows (expected %d, got %d)\n", ROWS, current_row);
        cJSON_Delete(root);
        return 1;
    }

    cJSON_Delete(root); // 解析完成，释放 cJSON 对象

    // 6. 运行推理
    InferenceResult result;
    run_inference(sensor_data, &result);

    // 7. 打印结果到标准输出 (格式为 "xc,yc")
    //    使用足够的小数位数以进行精确比较
    printf("%.8f,%.8f\n", result.corrected_cog_x, result.corrected_cog_y);

    return 0; // 成功退出
}

/*

**如何使用:**

1.  **获取 cJSON**: 从 [cJSON GitHub](https://github.com/DaveGamble/cJSON) 下载 `cJSON.c` 和 `cJSON.h` 并放在项目目录中。
2.  **编译 C 代码**: 运行 `make`。这将生成 `inference_app` 可执行文件。
3.  **运行验证脚本**:
    ```bash
    python validate_c_inference.py 
    
*/