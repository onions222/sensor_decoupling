# Makefile
.PHONY: all lib verify verify_debug infer clean

# --- 配置 ---
CC = gcc
# CFLAGS for PC verification (启用所有警告, C11, O3, FPU math)
CFLAGS_BASE = -std=c11 -Wall -Wextra -g
CFLAGS_RELEASE = $(CFLAGS_BASE) -O3
CFLAGS_DEBUG = $(CFLAGS_BASE) -O0 -DDUMP_INTERMEDIATES # Add flag and disable optimization for easier debugging

# Default flags
CFLAGS ?= $(CFLAGS_RELEASE)

# CFLAGS for MCU (示例: arm-none-eabi-gcc)
# CFLAGS_MCU = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
# CFLAGS += $(CFLAGS_MCU)

# Includes
INCLUDES = -Iinclude -Isrc

# Linker (需要 math.h for float wrappers, and potentially std C lib for file IO in debug)
LDFLAGS = -lm

# --- 目录 ---
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
TESTS_DIR = tests
INTERMEDIATE_DIR = c_intermediate_outputs # Directory for C dumps

#
# --- C 源文件 (核心库) ---
SRCS = \
    $(SRC_DIR)/inference.c \
    $(SRC_DIR)/inference_core.c \
    $(SRC_DIR)/weights.c

OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# --- C 源文件 (测试程序) ---
TEST_SRCS = \
    $(TESTS_DIR)/verify.c \
    $(TESTS_DIR)/test_helpers.c

TEST_OBJS = $(patsubst $(TESTS_DIR)/%.c, $(OBJ_DIR)/test_%.o, $(TEST_SRCS))

# --- 目标 ---
TARGET_LIB = $(BUILD_DIR)/libinference.a
TARGET_VERIFY = $(BUILD_DIR)/verify
TARGET_INFER = $(BUILD_DIR)/infer

# --- 规则 ---

all: lib

lib: $(TARGET_LIB)

# Standard verification target (Release mode)
verify: CFLAGS = $(CFLAGS_RELEASE)
verify: $(TARGET_VERIFY)

# Debug verification target (Debug mode with intermediate dumps enabled)
verify_debug: CFLAGS = $(CFLAGS_DEBUG)
verify_debug: $(TARGET_VERIFY)
	@echo "Built verify program with intermediate dumping enabled."

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	@echo "AR $@"
	@ar rcs $@ $(OBJS)

$(TARGET_VERIFY): $(OBJS) $(TEST_OBJS)
	@mkdir -p $(BUILD_DIR)
	@echo "CC $@"
	@$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(TEST_OBJS) $(LDFLAGS)

# Simple inference CLI (float-in -> float-out)
TOOLS_DIR = tools
TOOLS_SRCS = $(TOOLS_DIR)/infer.c
TOOLS_OBJS = $(patsubst $(TOOLS_DIR)/%.c, $(OBJ_DIR)/tool_%.o, $(TOOLS_SRCS))

infer: CFLAGS = $(CFLAGS_RELEASE)
infer: $(TARGET_INFER)

$(TARGET_INFER): $(OBJS) $(TOOLS_OBJS)
	@mkdir -p $(BUILD_DIR)
	@echo "CC $@"
	@$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(TOOLS_OBJS) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/inference_core.h $(SRC_DIR)/inference.h $(SRC_DIR)/model_meta.h $(SRC_DIR)/weights.h include/config.h
	@mkdir -p $(OBJ_DIR)
	@echo "CC $< (CFLAGS=$(CFLAGS))"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%.o: $(TESTS_DIR)/%.c $(TESTS_DIR)/test_helpers.h $(SRC_DIR)/inference.h $(SRC_DIR)/inference_core.h include/config.h $(SRC_DIR)/model_meta.h
	@mkdir -p $(OBJ_DIR)
	@echo "CC $< (CFLAGS=$(CFLAGS))"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/tool_%.o: $(TOOLS_DIR)/%.c $(SRC_DIR)/inference.h $(SRC_DIR)/inference_core.h include/config.h $(SRC_DIR)/model_meta.h
	@mkdir -p $(OBJ_DIR)
	@echo "CC $< (CFLAGS=$(CFLAGS))"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR) $(INTERMEDIATE_DIR) verify_data/*.bin # Clean generated data too

