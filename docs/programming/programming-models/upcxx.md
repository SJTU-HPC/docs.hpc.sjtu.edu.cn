# UPC++

UPC++ exposes a PGAS memory model, including one-sided communication
(RMA and RPC). However, there are two major changes. These changes
reflect a design philosophy that encourages the UPC++ programmer to
directly express what can be implemented efficiently (ie without a
need for parallel compiler analysis).

* Most operations are non-blocking, and the powerful synchronization
  mechanisms encourage applications to design for aggressive
  asynchrony.
* All communication is explicit - there is no implicit data motion.

## Using UPC++ at NERSC

Up-to-date documentation for UPC++ is maintained on
the
[bitbucket wiki](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) of
the project.

To see what versions are available (generally the default is
recommended).

```shell
nersc$ module avail upcxx
```

## Example Makefile

!!! note
	A module must be loaded for this Makefile to work.

```Makefile
ifeq ($(wildcard $(UPCXX_INSTALL)/bin/upcxx-meta),)
$(error Please set UPCXX_INSTALL=/path/to/upcxx/install)
endif

EXTRA_FLAGS=-std=c++11 -g
CXX=$(shell $(UPCXX_INSTALL)/bin/upcxx-meta CXX)
PPFLAGS=$(shell $(UPCXX_INSTALL)/bin/upcxx-meta PPFLAGS) $(EXTRA_FLAGS)
LDFLAGS=$(shell $(UPCXX_INSTALL)/bin/upcxx-meta LDFLAGS) $(EXTRA_FLAGS)
LIBFLAGS=$(shell $(UPCXX_INSTALL)/bin/upcxx-meta LIBFLAGS)

all: main

.cpp: *.h *.hpp
	$(CXX) $< $(PPFLAGS) $(LDFLAGS) $(LIBFLAGS) -o $@

clean:
	rm -rf main

.PHONY: force
```
