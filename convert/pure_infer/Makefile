# Makefile for Quantized Inference Example

# --- Configuration ---
CC = gcc
# CFLAGS: Enable warnings, use C11 standard, optimize for size/speed (-O2)
CFLAGS_BASE = -std=c11 -Wall -Wextra -O2
# Add include paths for headers in 'include' and 'src' directories
INCLUDES = -Iinclude -Isrc
# Linker flags: Need math library for floorf, tanhf
LDFLAGS = -lm

# Default CFLAGS
CFLAGS = $(CFLAGS_BASE) $(INCLUDES)

# --- Directories ---
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# --- Source Files ---
# Source files required for the example executable
SRCS = \
	main_example.c \
	$(SRC_DIR)/inference_core.c \
	$(SRC_DIR)/inference_quantized.c \
	$(SRC_DIR)/weights.c

# Generate object file names based on source files
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(notdir $(SRCS)))

# --- Target ---
TARGET_EXEC = $(BUILD_DIR)/inference_example

# --- Rules ---

# Default target: build the executable
all: $(TARGET_EXEC)

# Rule to link the executable
$(TARGET_EXEC): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	@echo "LD $@"
	@$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile C source files from the root directory (main_example.c)
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile C source files from the src directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Target to run the compiled example
run: $(TARGET_EXEC)
	@echo "Running inference example..."
	@./$(TARGET_EXEC)

# Target to clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

# Phony targets (targets that don't represent files)
.PHONY: all run clean
