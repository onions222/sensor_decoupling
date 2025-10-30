# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -I. # -I. tells GCC to look for headers in the current dir
LDFLAGS = -lm # Link math library

# Source files
SRCS = main.c inference.c model_weights.c cJSON.c
# Object files (derived from SRCS)
OBJS = $(SRCS:.c=.o)
# Executable name
TARGET = inference_app

# Default rule: build the executable
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Rule to compile .c files into .o files
%.o: %.c inference.h model_weights.h cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
