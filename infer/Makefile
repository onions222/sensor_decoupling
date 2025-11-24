CC = gcc
CFLAGS = -std=c99 -O2
SRCS = main.c inference.c weights.c input_sample.c
OBJS = $(SRCS:.c=.o)
TARGET = infer_demo

.PHONY: all clean run py compare

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

py:
	python3 compare_inference.py

compare: $(TARGET)
	python3 compare_inference.py

clean:
	rm -f $(TARGET) $(OBJS)
