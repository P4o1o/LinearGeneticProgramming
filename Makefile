CC = gcc
CFLAGS = -O3 -Wall -pedantic -std=c2x -mavx512f -mavx512dq -msse2
DFLAGS = -ggdb3 -fsanitize=undefined
LIBFLAGS = -lm -fopenmp
SRCDIR = src
BINDIR = bin
ASMDIR = assembly
SRCFILES = $(wildcard $(SRCDIR)/*.c)
OBJFILES = $(patsubst $(SRCDIR)/%.c,$(BINDIR)/%.o,$(SRCFILES))
ASMFILES = $(patsubst $(SRCDIR)/%.c,$(ASMDIR)/%.s,$(SRCFILES))

all: lgp

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
