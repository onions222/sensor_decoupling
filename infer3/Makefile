# ============================================================
#  Makefile for C-based int8 inference (teacher model)
# ============================================================

CC      := gcc
CFLAGS  := -O3 -Wall -std=c11
LDFLAGS := -lm

TARGET  := test_infer

SRCS := \
    main.c \
    teacher_int8_infer.c

OBJS := $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean run