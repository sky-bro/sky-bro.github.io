Easily compile and use multiple glibc on a single machine. 

<!--more-->

## Get source and compile

I'll store my source at `~/src/glibc/`

you can download sources of glibc from [hit mirror](http://mirrors.hit.edu.cn/gnu/glibc/), or [tsinghua tuna](https://mirrors.tuna.tsinghua.edu.cn/gnu/glibc/) or [official](https://ftp.gnu.org/gnu/glibc/)

here take glibc-2.23 as an example, after downloading the source

```shell
cd ~/src/glibc
mkdir glibc-2.23-{build,out}
cd glibc-2.23-build
# -Wno-error will override -Werror which will make compilation fail on warnings
# -O3, if you specify CFLAGS, you must give it -Ox (glibc cannot be compiled without optimization: https://gnu.org/software/libc/manual/html_mono/libc.html#toc-Installing-the-GNU-C-Library)
# -g will generate embed debug information
../glibc-2.23/configure --prefix=~/src/glibc/glibc-2.23-out CFLAGS="-Wno-error -O3 -g" # this is important
# make # compile with single thread, slow: took me about 14 minutes
make -j`nproc` # faster: took me under 2 minutes
# make -j$((`nproc`+1)) # https://unix.stackexchange.com/questions/208568/how-to-determine-the-maximum-number-to-pass-to-make-j-option
make install # install to what you have configured in the `--prefix` (so mine will be installed to ~/src/glibc/glibc-2.23-out)
```

## use mgcc to compile programs

### mgcc

mgcc is just a shell wrapper that wraps the gcc command, you can compile programs same way as you would in gcc.
you can also specify `--glibc_install` with the installation path of some version of glibc to switch between different glibc's: `mgcc --glibc_install ~/src/glibc/glibc-2.32-out -g -o main main.c`

You can find my most up to date mgcc at [.dotfiles/bin/mgcc](https://github.com/sky-bro/.dotfiles/blob/master/bin/mgcc)

`vim ~/bin/mgcc`

```shell
#!/bin/sh

glibc_install=~/src/glibc/glibc-2.23-out

print_help() {
  echo "$0 [--glibc_install GLIBC_INSTALL_PATH] gcc_args"
}

while [[ $# -ge 1 ]]
do
  key="$1"
  case $key in
    --glibc_install)
      glibc_install="$2"
      shift
      ;;
    -h)
      print_help
      exit 0
      ;;
    *)
      others+=("$1")
      ;;
  esac
  shift
done

gcc \
  -L "${glibc_install}/lib" \
  -I "${glibc_install}/include" \
  -Wl,--rpath="${glibc_install}/lib" \
  -Wl,--dynamic-linker="${glibc_install}/lib/ld-linux-x86-64.so.2" \
  "${others[@]}"
```

### run & debug

`vim main.c`

```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    void *p1 = malloc(0x70);
    void *p2 = malloc(0x70);
    free(p1);
    free(p2);
    free(p1);
    return 0;
}
```

you can check if your mgcc works after compile `mgcc -g -o main main.c`:

```shell
ldd main
# linux-vdso.so.1 (0x00007ffd36f9a000)
# libc.so.6 => /home/sky/src/glibc/glibc-2.23-out/lib/libc.so.6 (0x00007fac09658000)
# /home/sky/src/glibc/glibc-2.23-out/lib/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2 (0x00007fac0981b000)
```

tell gdb where to look for source code: `dir ~/src/glibc/glibc-2.23/malloc`

```shell
gdb main
# ...
pwndbg> dir ~/src/glibc/glibc-2.23/malloc/
# ...
```

{{< figure src="./images/gdb malloc.png" caption="you can see malloc source code in gdb" alt="you can see malloc source code in gdb" >}}

### completion

if you are using zsh and oh-my-zsh (you should), you can add a line: `compdef mgcc=gcc` to your `~/.zshrc` file
this will allow mgcc to use gcc's completion

## run others' program with your libc

### CLI

```shell
LD_PRELOAD=./libc.so ./program
# or https://pullp.github.io/2020/11/06/11-glibc-basics/#2-2-run-with-specific-glibc
LD_PRELOAD=./libc.so ./ld.so ./program
```

### pwntools

```python
#coding:utf-8
from pwn import *

io = process("./program", env={"LD_PRELOAD":"./libc.so"})
io = process(["./ld.so", "./program"], env={"LD_PRELOAD":"./libc.so"})
```

## refs

* [glibc相关操作记录](https://pullp.github.io/2020/11/06/11-glibc-basics)
* [How to determine the maximum number to pass to make -j option?](https://unix.stackexchange.com/questions/208568/how-to-determine-the-maximum-number-to-pass-to-make-j-option)
* [offcial: glibc installation guide](https://gnu.org/software/libc/manual/html_mono/libc.html#toc-Installing-the-GNU-C-Library)
* [How can I link to a specific glibc version?](https://stackoverflow.com/questions/2856438/how-can-i-link-to-a-specific-glibc-version)
