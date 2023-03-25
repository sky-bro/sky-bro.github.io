
## Old way {#old-way}

use `rand()`, usually pair with a random initialization of the seed:

```c
srand(int(time(0))); // initialize the seed
rand(); // get a random int, [0, RAND_MAX]
```

get random int in `[0,x)`: `rand()%x`
get random real in `[0, 1]`: `rand()/double(RAND_MAX)`


## Modern way {#modern-way}


### Generators {#generators}


#### random_device {#random-device}

random_device is a uniformly-distributed integer random number generator that produces non-deterministic random numbers.
A good implementation should has its randomness come from a non-deterministic source (e.g. a hardware device).
We can just use this to get our random numbers, but it might come with a light performance price. A PRNG is much better


#### pseudo RNG {#pseudo-rng}

So, we usually use random_device to get our first random number, then use it to initialize other PRNGs, then we can more quickly get many more random numbers.
The STL implements several PRNGs (see [cppref: Predefined random number generators](https://en.cppreference.com/w/cpp/numeric/random)), not just the **Mersenne Twister** shown above, you can check


### Distributions {#distributions}

```c++
// ...
#include <random>

std::random_device rd; // get a seed
std::mt19937 g(rd());
```


## Use cases of PRNG {#use-cases-of-prng}

TODO...


## Refs {#refs}

-   [cppref: std::random_device](<https://en.cppreference.com/w/cpp/numeric/random/random_device>)
-   [cppref: Predefined random number generators](<https://en.cppreference.com/w/cpp/numeric/random>)
-   [std::random_shuffle is deprecated in C++14](<https://meetingcpp.com/blog/items/stdrandom_shuffle-is-deprecated.html>)
