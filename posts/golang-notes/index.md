
## Basics {#basics}


### Packages {#packages}

every go programs is made up of packages, entry function: _func main()_

import statements could be grouped together using parenthesis ("factored"
import statements)

```go
import "fmt"
import "math"
```

same as:

```go
import (
  "fmt"
  "math"
)
```

Capitalized names are exported, when importing a package, you can only refer
its exported names.


### Basic Types {#basic-types}

```go
package main

import (
  "fmt"
  "math/cmplx"
)

// bool

// string

// int int8 int16 int32 int 64
// uint uint8 uint16 uint32 uint64 unitptr

// bytes // alias for uint8

// rune // alias for int32, represents a unicode code point

// float32 float64

// complex64 complex128

var (
  ToBe   bool       = false
  MaxInt uint64     = 1<<64 - 1
  z      complex128 = cmplx.Sqrt(-5 + 12i)
)

func main() {
  fmt.Printf("Type: %T Value: %v\n", ToBe, ToBe)
  fmt.Printf("Type: %T Value: %v\n", MaxInt, MaxInt)
  fmt.Printf("Type: %T Value: %v\n", z, z)
}
```

the int, uint, and uintptr types are usually 32 bits on 32-bit systems and 64
bits on 64-bit systems.


### Variables {#variables}

_var_ statement declares a list of variables, the type comes last. A _var_
statement can be at package or function level.

A var declaration can include initializers, if an initializer is present, the
type can be omitted.

inside a function, the `:=` short assignment statement can be used instead of
a `var` declaration with implicit type

when the right hand side of the declaration is typed, the new variable is of
that same type.

with `const` keyword, you declare constants. Constants cannot be declared
using the `:=` syntax.

```go
// with initializer
var i, j int = 1, 2
// omit type (type inference)
var i, j = 1, 2
// short variable declarations (also type inference)
a := 3 // int
b := 3.14 // float64
c := 0.1 + 0.2i

// constants
const Pi = 3.14
// numeric constants are high-precision values
const (
  Big = 1 << 100    // 1 << 100
  Small = Big >> 99 // 2
)
```


### Functions {#functions}

```go
// type comes after variable name.
func add(x int, y int) int {
  return x + y;
}

// consecutive named function parameters share a type
func add(x, y int) int {
  return x + y;
}

// function can return any number of results
func swap(x, y string) (string, string) {
  return y, x
}

// naked return: a return without arguments, can harm readability in longer
// functions
func split(sum int) (x, y int) {
  x = sum * 4 / 9
  y = sum - x
  return
}
```


#### Methods {#methods}

Go has no classes, but you can define methods on types. A method is a
function with a special _receiver_ argument.

```go
package main

import (
  "fmt"
  "math"
)

type Vertex struct {
  X, Y float64
}

func (v Vertex) Abs() float64 {
  return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v *Vertex) Scale(f float64) {
  v.X = v.X * f
  v.Y = v.Y * f
}

func main() {
  v := Vertex{3, 4}
  v.Scale(10)
  fmt.Println(v.Abs()) // 50
}
```

****The method and its receiver type must be defined in the same package.****

```go
type MyFloat float64 // define your own float64 type
```


#### Pointer Receivers or Arguments {#pointer-receivers-or-arguments}

Receiver is just another argument, it can be nil.

If you wanna modify or not copying an argument, you should pass pointer
types.

In general, all methods on a given type should have either value or pointer
receivers, but not a mixture of both.


### Type conversions {#type-conversions}

`T(v)` converts the value `v` to the type `T`.

```go
i := 42
f := float64(i)
u := uint(f)
```

****no implicit conversion in go****


## Control FLow {#control-flow}


### For {#for}

Go only has for loop, no while loop, has three components:

-   init statement
-   condition statement
-   post statement

no parentheses, but curly braces `{}` are always required.

```go
package main

import "fmt"

func main() {
  sum := 0
  for i := 0; i < 10; i++ {
    sum += i
  }
  fmt.Println(sum)
}
```

any of the three components can be omitted, if you only have condition or
nothing left, semicolon can be omitted.

```go
sum := 0
// just like while loop
for sum < 1000 {
  sum++
}

for {
  // loop forever
}
```

you can `continue` or `break` inside a loop


### If {#if}

like `for` loops, no parentheses `()`, but curly braces `{}` are required

can have init statement

```go
if i := 0; i != 0 {
  fmt.Println("what?")
} else if i == 0 {
  fmt.Println("got 0")
} else {
  fmt.Println("no way here")
}
```


### Switch {#switch}

-   shorter way to write a sequence of `if-else` statements.

-   no break or default fallthrough in switch

-   cases need not be constants

-   values need not to be integers (but types must match)

-   also support init statements like `if` and `for`.

<!--listend-->

```go
package main

import (
  "fmt"
  "runtime"
)

func main() {
  fmt.Print("Go runs on ")
  switch os := runtime.GOOS; os {
  case "darwin":
    fmt.Println("OS X.")
  case "linux":
    fmt.Println("Linux.")
  default:
    // freebsd, openbsd,
    // plan9, windows...
    fmt.Printf("%s.\n", os)
  }
}
```

-   you can use `fallthrough` keyword to fallthrough
-   mutiple statements in a single case:
-   omit condition is the same as `switch true` (clean way to write long
    if-then-else chains)

<!--listend-->

```go
package main

import (
  "fmt"
  "time"
)

func main() {
  t := time.Now()

  // same as: switch true
  switch {
  // default will always be last evaluated
  default:
    fmt.Println("default")
  case t.Hour() < 12:
    fmt.Println("morning")
  case t.Hour() < 17, true: // like useing '||', match any
    fmt.Println("afternoon")
    fallthrough
  case false:
    // even condition is false, fallthrough do fallthrough here
    fmt.Println("Are you ok?")
    // cannot put fallthrough in the last case or default
    // fallthrough
  }
}
```


### Defer {#defer}

a `defer` statement defers the execution of a function until the surrounding
function returns.

arguments evaluated immediately, but function call is not executed until the
surrounding function returns.

defered function calls are pushed onto a stack, so executed in last-in-first-outo order

```go
package main

import "fmt"

func main() {
  fmt.Println("counting")

  for i := 0; i < 10; i++ {
    defer fmt.Println(i)
  }

  fmt.Println("done")
}
```


## Advanced types {#advanced-types}


### Pointers {#pointers}

A pointer holds the memory address of a value.

Go has no pointer arithmetic.

```go
var p *int
i := 42
p = &i // referencing
fmt.Println(*p) // dereferencing
```


### Structs {#structs}

can be defined inside functions

access struct fileds using a dot

struct fields can also be accessed through a struct pointer, without explicit
dereferencing.

```go
package main

import "fmt"

func main() {
  type Vertex struct {
    X int
    Y int
  }
  v := Vertex{1, 2}
  p := &v // pointer to a struct
  v.X = 4
  (*p).Y = 6 // dereference the struct first - cumbersum
  p.Y = 5    // without explicit dereference
  fmt.Println(v)
}
```

struct literal

```go
package main

import "fmt"

type Vertex struct {
  X, Y int
}

func main() {
  var (
    v1 = Vertex{1, 2}  // {1, 2}, has type Vertex
    v2 = Vertex{Y: 1}  // {0, 1}
    v3 = Vertex{}      // {0, 0}
    p  = &Vertex{3, 4} // has type *Vertex
  )

  // {1 2} {0 1} {0 0} &{3 4}
  fmt.Println(v1, v2, v3, p)
}
```


### Arrays {#arrays}

`[n]T` is an array of `n` values of type `T`.

Arrays cannot be resized

```go
package main

import "fmt"

func main() {
  // [1 2 0]
  fmt.Println([3]int{1, 2})
}
```


### Slices {#slices}

A slice is a dynamically-sezed, flexible view into the elements of an array.

`[n]T` is an array of type T and length n.

`[]T` is a slice of type T, it does not store any data, it just describes a
section of the underlying array.

```go
package main

import "fmt"

func main() {
  // array literal
  v := [5]int{1, 2}
  // slice literal
  // v := []int{1, 2, 0, 0, 0}
  v1 := v[1:3]
  fmt.Println(cap(v))  // 5
  fmt.Println(cap(v1)) // 4, counting from first element in the slice
  fmt.Println(len(v1)) // 2
  fmt.Println(v1)      // [2 0]
}
```

When slicing, you may omit the high or low bounds to use their defaults
instead (_0_ for low, _len_ for high)

making a slice

```go
a := make([]int, 5)    // len(a)=5, cap(a)=5
b := make([]int, 0, 5) // len(b)=0, cap(b)=5
b = b[:cap(b)]         // len(b)=5, cap(b)=5
b = b[1:]              // len(b)=4, cap(b)=4
```

appending to a slice

if the backing array is to small to fit all the given values a bigger array
will be allocated. The returned slice will point to the newly allocated
array.

range

```go
package main

import "fmt"

var pow = []int{1, 2, 4, 8, 16, 32, 64, 128}

func main() {
  // for i, _ := range pow
  // for i := range pow
  // for _, v := range pow
  for i, v := range pow {
    fmt.Printf("2**%d = %d\n", i, v)
  }
}
```


### Maps {#maps}

```go
package main

import "fmt"

type Vertex struct {
  X, Y int
}

func main() {
  var v = map[int]Vertex{
    3: {1, 2},
  }
  // x := v[2]
  x, ok := v[2]
  // ok is false
  if !ok {
    fmt.Println("no v[2]")
    fmt.Printf("x is the zero value of Vertex, which is %v\n", x)
  }
  fmt.Println(v[3])
  // insert or update an element
  v[3] = Vertex{3, 4}
  // delete a key
  delete(v, 3)
}
```


### Function Type {#function-type}

Functions are values too, they can be used as funtion arguments and return
values.

```go
package main

import "fmt"

func f(fn func(int) string, x int) string {
  return fn(x)
}

func main() {

  myF := func(x int) string {
    return "xy"
  }

  fmt.Println(f(myF, 3))
}
```

Receiver is actually the first argument of a method:

```go
package main

import (
  "fmt"
  "math"
)

type Vertex struct {
  X, Y float64
}

func (v Vertex) Abs() float64 {
  return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v *Vertex) Scale(f float64) {
  v.X = v.X * f
  v.Y = v.Y * f
}

func f(fn func(Vertex) float64, v *Vertex) {
  fmt.Println(fn(*v))
}

func f2(fn func(*Vertex) float64, v *Vertex) {
  fmt.Println(fn(v))
}

func main() {
  v := Vertex{3, 4}
  f(Vertex.Abs, &v)
  f2((*Vertex).Abs, &v)
  // these two are different function
  // receiver is actually the first argument of method
  fmt.Printf("%T\n", (*Vertex).Scale)
  fmt.Printf("%T\n", v.Scale)
}
```

A closure is a function value that references variables from outside its
body.

```go
package main

import "fmt"

func adder() func(int) int {
  sum := 0
  return func(x int) int {
    sum += x
    return sum
  }
}

func main() {
  pos, neg := adder(), adder()
  for i := 0; i < 10; i++ {
    fmt.Println(
      pos(i),
      neg(-2*i),
    )
  }
}
```


### Interfaces {#interfaces}

An interface type is a set of method signatures.

An interface value is any type that has implemented those methods
(implemented implicitly, no "implements" keyword).

```go
package main

import (
  "fmt"
  "math"
)

type Abser interface {
  Abs() float64
}

func main() {
  var a Abser
  f := MyFloat(-math.Sqrt2)
  v := Vertex{3, 4}

  a = f  // a MyFloat implements Abser
  a = &v // a *Vertex implements Abser

  // In the following line, v is a Vertex (not *Vertex)
  // and does NOT implement Abser.
  // a = v

  fmt.Println(a.Abs())
}

type MyFloat float64

func (f MyFloat) Abs() float64 {
  if f < 0 {
    return float64(-f)
  }
  return float64(f)
}

type Vertex struct {
  X, Y float64
}

func (v *Vertex) Abs() float64 {
  return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
```

printing value and type of an interface is the same as printing its
underlying value and type.

```go
package main

import "fmt"

type I interface {
  M()
}

type T struct {
  S string
}

func (t *T) M() {
  if t == nil {
    fmt.Println("<nil>")
    return
  }
  fmt.Println(t.S)
}

func main() {
  var i I

  var t *T
  i = t
  describe(i)
  i.M()

  i = &T{"hello"}
  describe(i)
  i.M()
}

func describe(i I) {
  fmt.Printf("(%v, %T)\n", i, i)
}
```

The interface that specifies zero methods is known as the empty interface.

```go
package main

import "fmt"

func main() {
  var i interface{}
  describe(i)

  i = 42
  describe(i)

  i = "hello"
  describe(i)
}

func describe(i interface{}) {
  fmt.Printf("(%v, %T)\n", i, i)
}
```


#### Type assertions {#type-assertions}

try converting an interface to its underlying value of type T: `s := i.(T)`

here `T` must implement methods of i.

```go
package main

import "fmt"

func main() {
  var i interface{} = "hello"

  s := i.(string)
  fmt.Println(s)

  s, ok := i.(string)
  fmt.Println(s, ok)

  f, ok := i.(float64)
  fmt.Println(f, ok)

  f = i.(float64) // panic
  fmt.Println(f)
}
```

```go
package main

import "fmt"

type Vertex struct {
  x, y int
}

// value of type *Vertex can also call method M()
func (Vertex) M() {}

func main() {
  var v Vertex

  var i interface {
    M()
  } = v

  // argument here must implement methods of the interface
  t, ok := i.(*Vertex)
  des(t) // *main.Vertex, <nil>
  chk(t, ok)

  t2, ok := i.(Vertex)
  des(t2) // main.Vertex, {0, 0}
  chk(t2, ok)
}

func des(v interface{}) {
  fmt.Printf("%T, %v\n", v, v)
}

func chk(t interface{}, ok bool) {
  if !ok {
    fmt.Println("type not correct, zero value returned:", t)
  } else {
    fmt.Println("type correct, value is:", t)
  }
}
```


#### type switches {#type-switches}

```go
package main

import "fmt"

func do(i interface{}) {
  switch v := i.(type) {
  case int:
    fmt.Printf("Twice %v is %v\n", v, v*2)
  case string:
    fmt.Printf("%q is %v bytes long\n", v, len(v))
  default:
    // here v has the same type as i
    fmt.Printf("I don't know about type %T!\n", v)
  }
}

func main() {
  do(21)
  do("hello")
  do(true)
}
```


## Zero values {#zero-values}

variables declared without an explicit initial value are given their zero
value.

-   0 for numeric types
-   false for the boolean type
-   "" (the empty string) for the strings
-   `{<default field values>}` for structs
-   nil for slice (len and cap of a nil slice is 0)
-   nil for interfaces
-   nil for pointers


## Common Interfaces {#common-interfaces}


### Error {#error}

When `fmt` prints values, it looks for the error interface first:

```go
type error interface {
  Error() string
}
```

if the interface value is not `<nil>`, the Error() method will be invoked by
`fmt` to get the error string.

```go
i, err := strconv.Atoi("42")
if err != nil {
  fmt.Printf("couldn't convert number: %v\n", err)
  return
}
fmt.Println("Converted integer:", i)
```

Do not print the interface value in the Error() method directly, it will cause
infinite loop.

```go
type ErrNegativeSqrt float64

func (e ErrNegativeSqrt) Error() string {
  // do not print e directly, infinite loop here
  // fmt.Println(e)
  return fmt.Sprintln("cannot Sqrt negative number: ", float64(e))
}
```


### Reader {#reader}

the `io.Reader` interface has a `Read` method:

```go
func (T) Read(b []byte) (n int, err error)
```

read populates the given byte slice with data and returns the number of bytes
populated and an error value.

it returns an `io.EOF` error when the stream ends.

```go
package main

import (
  "fmt"
  "io"
  "strings"
)

func main() {
  r := strings.NewReader("Hello, Reader!")

  b := make([]byte, 8)
  for {
    n, err := r.Read(b)
    fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
    fmt.Printf("b[:n] = %q\n", b[:n])
    if err == io.EOF {
      break
    }
  }
}
```


### Image {#image}

`image.Image` defines the Image interface

```go
type Image interface {
  ColorModel() color.Model
  Bounds() Rectangle
  At(x, y int) color.Color
}
```


## Goroutines {#goroutines}

a _goroutine_ is a lightweight thread managed by the Go runtime.


### Channels {#channels}

By default, sends and receives block until the other side is ready. This
allows goroutines to synchronize without explicit locks or condition
variables.

Channels aren't like files, you don't usually need to close them. Closing is
only necessary when the receiver must be told there are no more values
coming, such as to terminate a `range` loop.

```go
package main

import "fmt"

func fib(n int, c chan int) {
  a, b := 0, 1
  for i := 0; i < n; i++ {
    c <- a
    a, b = b, a+b
  }
  close(c)
}

func main() {
  c := make(chan int)
  go fib(10, c)
  for x := range c {
    fmt.Println(x)
  }
  // "ok" is false if there are:
  // 1. no more values to receive
  // 2. and the channel is closed
  // x, ok := <- c
}
```


### Select {#select}

_select_ statement lets a goroutine wait on multiple communication
operations.

_select_ blocks until one of its cases can run (by adding a _default_ case,
it won't block). It chooses one at random if multiple are ready.

```go
package main

import "fmt"

func fibonacci(c, quit chan int) {
  x, y := 0, 1
  for {
    select {
    case c <- x:
      x, y = y, x+y
    case <-quit:
      fmt.Println("quit")
      return
    }
  }
}

func main() {
  c := make(chan int)
  quit := make(chan int)
  go func() {
    for i := 0; i < 10; i++ {
      fmt.Println(<-c)
    }
    quit <- 0
  }()
  fibonacci(c, quit)
}
```


### Mutex {#mutex}

_sync.Mutex_ provides two methods: `Lock` and `Unlock`

```go
// SafeCounter is safe to use concurrently.
type SafeCounter struct {
  mu sync.Mutex
  v  map[string]int
}

func (c *SafeCounter) Value(key string) int {
  c.mu.Lock()
  // Lock so only one goroutine at a time can access the map c.v.
  defer c.mu.Unlock()
  return c.v[key]
}
```


## Practice {#practice}


### Sqrt {#sqrt}

```go
package main

import "fmt"

func Sqrt(x float64) (res float64) {
  res = 1.
  diff := 1.
  for diff > 1e-5 || diff < -1e-5 {
    diff = (res*res - x) / (2 * res)
    res -= diff
  }
  return
}

func main() {
  fmt.Println(Sqrt(4))
}
```


### WordCount {#wordcount}

```go
package main

import (
  "strings"

  "golang.org/x/tour/wc"
)

func WordCount(s string) (m map[string]int) {
  m = make(map[string]int)
  for _, x := range strings.Fields(s) {
    m[x]++
  }
  return m
}

func main() {
  wc.Test(WordCount)
}
```


### Fibonacci closure {#fibonacci-closure}

```go
package main

import "fmt"

// fibonacci is a function that returns
// a function that returns an int.
func fibonacci() func() int {
  a, b := 0, 1
  return func() int {
    ret := a
    a, b = b, a + b
    return ret
  }
}

func main() {
  f := fibonacci()
  for i := 0; i < 10; i++ {
    fmt.Println(f())
  }
}
```


### Sqrt with Error Handling {#sqrt-with-error-handling}

```go
package main

import (
  "fmt"
  "math"
)

type ErrNegativeSqrt float64

func (e ErrNegativeSqrt) Error() string {
  // float64(e) here is important
  // fmt.Sprint(e) will cause infinite loop!
  return fmt.Sprint("cannot Sqrt negative number:", float64(e))
}

func Sqrt(x float64) (float64, error) {
  if x < 0 {
    return x, ErrNegativeSqrt(x)
  }
  return math.Sqrt(x), nil
}

func main() {
  fmt.Println(Sqrt(2))
  fmt.Println(Sqrt(-2))
}
```


### rot13Reader {#rot13reader}

```go
package main

import (
  "io"
  "os"
  "strings"
)

type rot13Reader struct {
  r io.Reader
}

func (rot13reader rot13Reader) Read(b []byte) (int, error) {
  n, err := rot13reader.r.Read(b)
  if err != nil {
    return 0, io.EOF
  }
  for i := 0; i < n; i++ {
    switch c := b[i]; {
    case c >= 'A' && c <= 'Z':
      b[i] = 'A' + (b[i]-'A'+13)%26
    case c >= 'a' && c <= 'z':
      b[i] = 'a' + (b[i]-'a'+13)%26
    }
  }
  return n, nil
}

func main() {
  s := strings.NewReader("Lbh penpxrq gur pbqr!")
  r := rot13Reader{s}
  io.Copy(os.Stdout, &r)
}
```


### Implement Image interface {#implement-image-interface}

```go
package main

import (
  "image"
  "image/color"

  "golang.org/x/tour/pic"
)

type Image struct {
  w, h int
}

func (img Image) ColorModel() color.Model {
  return color.RGBAModel
}

func (img Image) Bounds() image.Rectangle {
  return image.Rect(0, 0, img.w, img.h)
}

func (img Image) At(x, y int) color.Color {
  return color.RGBA{uint8(x + y), uint8(x + y), 255, 255}
}

func main() {
  m := Image{100, 100}
  pic.ShowImage(m)
}
```


### Web Crawler {#web-crawler}

```go
package main

import (
  "fmt"
  "sync"
)

type Fetcher interface {
  // Fetch returns the body of URL and
  // a slice of URLs found on that page.
  Fetch(url string) (body string, urls []string, err error)
}

type url2Dep struct {
  mu sync.Mutex
  mp map[string]int
}

func (u *url2Dep) insertUrl(url string, dep int) {
  u.mu.Lock()
  defer u.mu.Unlock()
  u.mp[url] = dep
}

func (u *url2Dep) getDep(url string) (int, bool) {
  u.mu.Lock()
  defer u.mu.Unlock()
  dep, ok := u.mp[url]
  return dep, ok
}

// Crawl uses fetcher to recursively crawl
// pages starting with url, to a maximum of depth.
func Crawl(u *url2Dep, url string, depth int, fetcher Fetcher) {
  defer wg.Done()
  if depth <= 0 {
    return
  }
  if dep, ok := u.getDep(url); !ok || dep < depth {
    u.insertUrl(url, depth)
  } else {
    fmt.Printf("visited: %s\n", url)
    return
  }
  body, urls, err := fetcher.Fetch(url)
  if err != nil {
    fmt.Println(err)
    return
  }
  fmt.Printf("found: %s %q\n", url, body)
  for _, nextUrl := range urls {
    wg.Add(1)
    go Crawl(u, nextUrl, depth-1, fetcher)
  }
}

var wg sync.WaitGroup

func main() {
  u := url2Dep{mp: make(map[string]int)}
  wg.Add(1)
  go Crawl(&u, "https://golang.org/", 4, fetcher)
  wg.Wait()
}

// fakeFetcher is Fetcher that returns canned results.
type fakeFetcher map[string]*fakeResult

type fakeResult struct {
  body string
  urls []string
}

func (f fakeFetcher) Fetch(url string) (string, []string, error) {
  if res, ok := f[url]; ok {
    return res.body, res.urls, nil
  }
  return "", nil, fmt.Errorf("not found: %s", url)
}

// fetcher is a populated fakeFetcher.
var fetcher = fakeFetcher{
  "https://golang.org/": &fakeResult{
    "The Go Programming Language",
    []string{
      "https://golang.org/pkg/",
      "https://golang.org/cmd/",
    },
  },
  "https://golang.org/pkg/": &fakeResult{
    "Packages",
    []string{
      "https://golang.org/",
      "https://golang.org/cmd/",
      "https://golang.org/pkg/fmt/",
      "https://golang.org/pkg/os/",
    },
  },
  "https://golang.org/pkg/fmt/": &fakeResult{
    "Package fmt",
    []string{
      "https://golang.org/",
      "https://golang.org/pkg/",
    },
  },
  "https://golang.org/pkg/os/": &fakeResult{
    "Package os",
    []string{
      "https://golang.org/",
      "https://golang.org/pkg/",
    },
  },
}
```
