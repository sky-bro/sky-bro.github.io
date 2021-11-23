Smart pointer is a practice of RAII (Resource Acquisition is Initialization), we use a smart pointer to manage memory allocated on the heap. (here the "resource" is the allocated heap memory)

<!--more-->

## why better not use raw pointers

1. declaration doesn't indicate whether it points to an array or a single object (delete or delete [])
2. you don't know where or not you should delete it after using it (ownership is unclear)
3. doesn't know where you should `delete` it or `close` it (what's the right deleter)
4. what if you cannot delete it (exception occurred)
5. or mistakenly delete it multiple times (double free)
6. no way to tell if a pointer dangles (not set to 0 automatically after being deleted)

## auto_ptr (deprecated since c++11, removed since c++17)

now deprecated.

born in C++1998 (don't have move semantics back then), copying an auto_ptr sets it to null.
So there are many usage restrictions (you can only move it, cannot actually copy it), it was not possible to store auto_ptr in containers.

### implement an auto_ptr

```c++
template<typename T>
class my_autoptrtr {
private:
    T* ptr;
public:
    explicit my_autoptrtr(T* p): ptr(p) {}
    ~my_autoptrtr() { delete ptr; }
    // no move semantics, so copy constructer have to move other object -- wrong semantic
    explicit my_autoptrtr(my_autoptrtr& other): ptr(other.ptr) { other.ptr = nullptr; }
    my_autoptrtr<T>& operator=(my_autoptrtr<T>& other) {
        if (this != &other) { // no need to assign if same object
            // delete ptr;
            // ptr = other.ptr;
            // other.ptr = nullptr;
            reset(other.release());
        }
        return *this;
    }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    T* get() const { return ptr; }
    T* release() {
        T* p = ptr;
        ptr = nullptr;
        return p;
    }
    void reset(T* p = nullptr) {
        if (p != ptr) {
            delete ptr;
            ptr = p;
        }
    }
};
```

## unique_ptr

> Item 18: Use std::unique_ptr for exclusive-ownership resource management

unique_ptr is move-only (no copy constructor - deleted), because this ownership is unique / for only one object.

If you really need to transfer the ownership of this resource to another pointer, `move` it.

Usually the responsibility is simply 'to call delete'.

### Example usage

There're two situations:

* When making a pointer pointing to a single object, make_unique will call the suitable constructor based on the given arguments.
* When making a pointer pointing to an array of objects, make_unique will use the default constructor. (if not exists, compilation error)

```c++
unique_ptr<int> p1 = make_unique<int>(3); // 3
unique_ptr<int[]> p2 = make_unique<int[]>(3); // {?, ?, ?}
unique_ptr<vector<int>> p3 = make_unique<vector<int>>(3, 1); // {1, 1, 1}, use constructor vector<int>(3, 1)
unique_ptr<vector<int>[]> p4 = make_unique<vector<int>[]>(3); // {{}, {}, {}}, use default constructor
```

### Implement a unique_ptr (simplified)

```c++
template <typename T>
class my_unique_ptr {
 private:
  T* ptr;

 public:
  my_unique_ptr() : ptr(nullptr) {}
  explicit my_unique_ptr(T* p) : ptr(p) {}
  ~my_unique_ptr() {
    if (ptr) delete ptr;
  }

  my_unique_ptr(my_unique_ptr& other) = delete;
  my_unique_ptr<T>& operator=(my_unique_ptr<T>& other) = delete;

  my_unique_ptr(my_unique_ptr&& other): ptr(other.release()) {}
  my_unique_ptr<T>& operator=(my_unique_ptr<T>&& other) {
    reset(other.release());
    return *this;
  }

  T& operator*() const { return *ptr; }
  T* operator->() const { return ptr; }
  bool operator==(const my_unique_ptr<T>& other) { return ptr == other.ptr; }
  explicit operator bool() const { return ptr; }
  T* get() const { return ptr; }
  T* release() {
    T* p = ptr;
    ptr = nullptr;
    return p;
  }
  void reset(T* p = nullptr) {
    if (p != ptr) {
      delete ptr;
      ptr = p;
    }
  }
};
```

## shared_ptr

> Item

### weak_ptr

it allows you to locate an object if it’s still around, but doesn’t keep it around if nothing else needs it.

### enable_shared_from_this

```c++
class C: public enable_shared_from_this<C> {
    public:
    void print() {
        cout << "C::print" << endl;
    }
    ~C() {
        cout << "C destructed!" << endl;
    }
};

int main(int argc, char const *argv[])
{
    make_shared<C>()->print();
    cout << make_shared<C>()->shared_from_this().use_count() << endl; // 2
    cout << "done!" << endl;
}
```

### Implement a simple shared_ptr

```c++
template <typename T>
class my_shared_ptr {
 private:
  T* ptr;
  int* cnt; // reference count

 public:
  my_shared_ptr(T* p) : ptr(p), cnt(new int(1)) {}
  my_shared_ptr(const my_shared_ptr<T>& other)
      : ptr(other.ptr), cnt(other.cnt) {
    ++(*cnt);
  }
  my_shared_ptr<T>& operator=(const my_shared_ptr<T>& other) {
    if (ptr != other.ptr) {
      if (--(*cnt) == 0) {
        delete ptr;
        delete cnt;
      }
      ptr = other.ptr;
      cnt = other.cnt;
      ++(*cnt);
    }
    return *this;
  }
  T* operator->() { return ptr; }
  T& operator*() { return *ptr; }
  int getcnt() { return *cnt; }
  ~my_shared_ptr() {
    if (--(*cnt) == 0) {
      delete ptr;
      ptr = nullptr;
      delete cnt;
      cnt = nullptr;
    }
  }
};
```

## use make_\* to create smart pointers

## custom deleter

Smart Pointers allow you to write your own custom deleters (a function), instead of using the default deleter inside which uses `delete p;` or `delete [] p;`.

Using `make_*` to create smart pointers is a good practice, but when you use custom deleters, it means that you'll need to create the raw pointer yourself too.

This can be resolved by creating you own make functions, check the examples below:

> taken from ref: [shared_ptr and FILE for wrapping cstdio (update: also dlfcn.h)](https://codereview.stackexchange.com/questions/4679/shared-ptr-and-file-for-wrapping-cstdio-update-also-dlfcn-h)

a shared_ptr for `fopen()/fclose()`

```c++
std::shared_ptr<std::FILE> make_shared_file(const char * filename, const char * flags)
{
  std::FILE * const fp = std::fopen(filename, flags);
  // because we cannot fclose a null pointer
  return fp ? std::shared_ptr<std::FILE>(fp, std::fclose) : std::shared_ptr<std::FILE>();
}
```

a unique_ptr for `fopen()/fclose()` (unique_ptr won't call deleter on a null pointer)

```c++
typedef std::unique_ptr<std::FILE, int (*)(std::FILE *)> unique_file_ptr;
unique_file_ptr make_unique_file(const char * filename, const char * flags)
{
  return unique_file_ptr(std::fopen(filename, flags), std::fclose);
}
```

a unique_ptr for `dlopen()/dlclose()`

```c++
typedef std::unique_ptr<void,  int (*)(void *)> unique_library_ptr;

static unique_library_ptr make_library(const char * filename, int flags)
{
  return unique_library_ptr(dlopen(filename, flags), dlclose);
}
```

## inheritance

shared_ptr

```c++
class A {
  int x;
};

class A2 {
  int x;
};

class B : public A, public A2 {};

int main(int argc, char const *argv[]) {
  shared_ptr<B> ptrB = make_shared<B>();
  // 0x55b4e10ecf20
  cout << ptrB.get() << endl;
  shared_ptr<A2> ptrA = ptrB;
  // 0x55b4e10ecf24
  cout << ptrA.get() << endl;
  return 0;
}
```

unique_ptr

```c++
class A {
  int x;
};

class A2 {
  int x;
};

class B : public A, public A2 {};

int main(int argc, char const *argv[]) {
  unique_ptr<B> ptrB = make_unique<B>();
  cout << ptrB.get() << endl;
  // free(): invalid pointer
  // unique_ptr<A2> ptrA = move(ptrB);
  unique_ptr<A> ptrA = move(ptrB);
  cout << ptrA.get() << endl;
  return 0;
}
```

## Rules of thumb

* treat smart pointers just like raw pointer types
  * pass by value
  * return by value (of course)
  * passing a pointer by reference is usually a code smell, same goes for smart pointers

## Refs

* [shared_ptr and FILE for wrapping cstdio (update: also dlfcn.h)](https://codereview.stackexchange.com/questions/4679/shared-ptr-and-file-for-wrapping-cstdio-update-also-dlfcn-h)
* [Scott Meyers EMC++ Item 18](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/ch04.html)
* [CppCon 2019: Arthur O'Dwyer "Back to Basics: Smart Pointers"](https://www.youtube.com/watch?v=xGDLkt-jBJ4)
