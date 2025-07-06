CC = gcc
CLANG = clang
THREADS ?= 16
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c2x -DOMP_NUM_THREADS=$(THREADS) -msse2 -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq # -mavx512ifma -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq
DFLAGS = -ggdb3 -fsanitize=undefined -fsanitize=signed-integer-overflow -pg -g
LIBFLAGS = -lm -fopenmp
SRCDIR = src
BINDIR = bin
ASMDIR = assembly
SRCFILES = $(wildcard $(SRCDIR)/*.c)
OBJFILES = $(patsubst $(SRCDIR)/%.c,$(BINDIR)/%.o,$(SRCFILES))
ASMFILES = $(patsubst $(SRCDIR)/%.c,$(ASMDIR)/%.s,$(SRCFILES))

.PHONY: all gcc clang asm fast clean

all: LGP

fast: CC=gcc
fast: DFLAGS=
fast: LGP

gcc: CC=gcc
gcc: LGP

clang: CC=clang
clang: LGP

single: CFLAGS=-O3 -Wall -Wextra -pedantic -std=c2x
single: LGP

sse2: CFLAGS=-O3 -Wall -Wextra -pedantic -std=c2x -msse2
sse2: LGP

avx2: CFLAGS=-O3 -Wall -Wextra -pedantic -std=c2x -msse2 -mavx2
avx2: LGP

avx512: LGP

asm: $(ASMFILES)

LGP : $(OBJFILES)
	$(CC) $(CFLAGS) $(DFLAGS) -o LGP $(OBJFILES) $(LIBFLAGS)

# Python interface shared library
python: DFLAGS=
python: CFLAGS += -fPIC
python: liblgp.so

liblgp.so: $(filter-out $(BINDIR)/main.o,$(OBJFILES))
	$(CC) $(CFLAGS) -shared -fPIC -o liblgp.so $(filter-out $(BINDIR)/main.o,$(OBJFILES)) $(LIBFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.c | $(BINDIR)
	$(CC) $(CFLAGS) $(DFLAGS) -fPIC -c $< -o $@ $(LIBFLAGS)

$(ASMDIR)/%.s: $(SRCDIR)/%.c | $(ASMDIR)
	$(CC) $(CFLAGS) -S -masm=intel $< -o $@ $(LIBFLAGS)

$(BINDIR) $(ASMDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BINDIR)/*.o $(ASMDIR)/*.s LGP
