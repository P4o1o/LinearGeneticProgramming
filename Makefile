CC = gcc
CLANG = clang
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c2x -msse2 -mavx2 -mavx512f -mavx512dq -mavx512vl
DFLAGS = -ggdb3 -fsanitize=undefined -fsanitize=signed-integer-overflow
LIBFLAGS = -lm -fopenmp
SRCDIR = src
BINDIR = bin
ASMDIR = assembly
SRCFILES = $(wildcard $(SRCDIR)/*.c)
OBJFILES = $(patsubst $(SRCDIR)/%.c,$(BINDIR)/%.o,$(SRCFILES))
ASMFILES = $(patsubst $(SRCDIR)/%.c,$(ASMDIR)/%.s,$(SRCFILES))

.PHONY: all gcc clang asm fast clean

all: lgp

fast: CC = gcc
fast: DFLAGS=
fast: lgp

gcc: CC = gcc
gcc: lgp

clang: CC = clang
clang: lgp

asm: $(ASMFILES)

lgp : $(OBJFILES)
	$(CC) $(CFLAGS) $(DFLAGS) -o lgp $(OBJFILES) $(LIBFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@ $(LIBFLAGS)

$(ASMDIR)/%.s: $(SRCDIR)/%.c | $(ASMDIR)
	$(CC) $(CFLAGS) -S -masm=intel $< -o $@ $(LIBFLAGS)

$(BINDIR) $(ASMDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BINDIR)/*.o $(ASMDIR)/*.s lgp
