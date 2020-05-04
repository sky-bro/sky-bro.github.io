
## Yes, by default

I heard before that using `scanf/printf` is faster than using `cin/cout`, and it's true from my real experience, but I really didn't get to know the reason behind, and later in leetcode, I saw others include these lines in their code:

```C++
ios::sync_with_stdio(false);
cin.tie(NULL);
cout.tie(NULL);
```

I'm almost certain that these lines are included to speed up their code. So out of curiosity, I did some searching, here's what I've found:
<!-- more -->

## Why cin & cout is slow

### First Reason: unbuffered streams

By default, C++ streams are synchronized to the standard C streams after each input/output operation, meaning that C++ streams are unbuffered, each I/O operation on a C++ stream is immediately applied to the corresponding C stream's buffer.

So this is the main reaseon: **To be compatible with C (so you can mix your C code inside C++), C++ do not buffer its streams.**

### Second Reason: cin, cout streams are tied together

By default, `cin` is tied to `cout`, and `wcin` to `wcout`, guarantees the flushing of `cout` before `cin` executes an input. In pure C, you may have to guarantee this by manually using `fflush` after the `printf`.

## What these three lines do

### ios::sync_with_stdio(false)

As said above, by default c++ streams share c streams' buffers. This can be tweeked using the `ios::sync_with_stdio(bool sync = true)` function. After setting the sychronization to false, the synchronization between the C and C++ standard streams is disabled, C++ will use its own buffer.

### cin.tie(NULL) & cout.tie(NULL)

`tie` is used to tie a stream (in/out) to some output stream, if the parameter is `NULL`, untie this stream from any tied stream (returns previous tied stream)
if called without any argument, return the tied stream:

```c++
*cin.tie() << 123; // same as cout << 123; (by default, cin is tied to cout)
```

so this is just used to untie `cin/cout` from any tied output stream (normally we just need to use `cin.tie(0)`), then the previously tied output stream won't be forced to flush.

## When should I use the "speed up"

### Know the adventures

1. If you disable the synchronization, then C++ streams are allowed to have their own independent buffers, which makes mixing C- and C++-style I/O an adventure. Also, synchronized C++ streams are thread-safe (output from different threads may interleave, but you get no data races). (refer cppref)

2. If you untie `cin`, then later when you execute the lines below, you won't be guaranteed to see the prompt before the input request from `cin`.

```C++
string name;
cout << "Please input your name:";
cin >> name;
```

### Decide

If you do not mix use the C- and C++-style I/O, do not write multi-thread program, use `ios::sync_with_stdio(false)`.
If you do not care about seeing some output before your input, use `tie(0)`.

## Example

```C++
int main() {
  // ...
}

static int x = []() {ios::sync_with_stdio(false); cin.tie(0); return 0; } ();
```

## Refs

* [cppref: std::ios_base::sync_with_stdio](https://en.cppreference.com/w/cpp/io/ios_base/sync_with_stdio)
* [cppref: std::basic_ios<CharT,Traits>::tie](https://en.cppreference.com/w/cpp/io/basic_ios/tie)
* [Why do we need to tie std::cin and std::cout?](https://stackoverflow.com/questions/14052627/why-do-we-need-to-tie-stdcin-and-stdcout#14052757)
