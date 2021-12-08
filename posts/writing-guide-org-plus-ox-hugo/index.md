
I've recently switched to [org mode](https://orgmode.org/), now I write all my blogs in org mode ([blog-src/content-org/](https://github.com/sky-bro/blog-src/blob/master/content-org/)), and export them to `.md` files ([blog-src/content/](https://github.com/sky-bro/blog-src/blob/master/content/)) with ox-hugo.

So instead of editing `.md` files under `content` folder, now I write `.org` files stored under `content-org` folder.


## Create new post {#create-new-post}

Invoking org-capture-templates function, and choose hugo post template, as shown in Figure [1](#orga45c0a9)

<a id="orga45c0a9"></a>

{{< figure src="/images/posts/Writing-Guide-Org/org-capture-template-ox-hugo.gif" caption="Figure 1: creating new post with org-capture-template" >}}


## Front matter {#front-matter}

As in [ox-hugo: Custom Front-matter Parameters](https://ox-hugo.scripter.co/doc/custom-front-matter/), hugo front matters can be added like below:

```org
:PROPERTIES:
:EXPORT_HUGO_CUSTOM_FRONT_MATTER: :key1 value1 :key2 value2
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :key3 value3
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :key4 value4
:END:
```


## Code {#code}

Inline code with '=' or '~': `=echo 123=`, `~echo 456~`

Code block with

```org
#+begin_src c
  int main() {
    return 0
  }
#+end_src
```


## Images {#images}

Store all the images under `$HUGO_BASE_DIR/static/` folder (except some generated images), so just include them using relative path from the org file.

You can add caption and name (for referencing purpose: as in figure [2](#orgaa7cee8)) to an image.

<a id="orgaa7cee8"></a>

{{< figure src="/images/icons/gopher001.png" caption="Figure 2: Gogpher" >}}

```org
#+CAPTION: Gogpher
#+NAME: fig:gopher
[[../static/images/icons/gopher001.png]]
```


## Math Support (with MathJax) {#math-support--with-mathjax}

We need to have MathJax library in our front matter.

```org
:PROPERTIES:
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :libraries '(mathjax)
:END:
```

Inline formulas with `\$..\$`. This is inline math: \\(x^2 + y^2 = z^2 \frac{1}{2}\\).

Displayed equations with `\$\$..\$\$` or \\(\LaTeX\\) encironments. This is displayed math:

The code:

```tex
\begin{equation}\label{eq:1}
  \begin{split}
    a &= b+c-d\\
      &\quad +e-f\\
      &= g+h\\
      &= i
  \end{split}
\end{equation}
```

will be rendered as:

\begin{equation}\label{eq:1}
  \begin{split}
    a &= b+c-d\\\\
      &\quad +e-f\\\\
      &= g+h\\\\
      &= i
  \end{split}
\end{equation}

{{< alert theme="warning" >}}
It seems that zzo theme does not support math equation referencing and numbering yet?
{{< /alert >}}


## Diagrams {#diagrams}


### Plantuml {#plantuml}

{{< figure src="/images/posts/Writing-Guide-Org/first.svg" >}}


## revealjs / presentation {#revealjs-presentation}


## shortcodes {#shortcodes}


### Alert {#alert}

You can have alert like this:

```org
{{</* alert theme="info" dir="ltr" */>}}
theme could be one of: success, info, warning, danger
{{</* /alert */>}}
```

{{< alert theme="success" >}}
this is a success.
{{< /alert >}}

{{< alert theme="info" >}}
this is a info.
{{< /alert >}}

{{< alert theme="warning" >}}
this is a warning.
{{< /alert >}}

{{< alert theme="danger" >}}
this is a danger.
{{< /alert >}}


### Notice {#notice}

```org
{{</* notice success "This is a success type of notice" */>}}
notice could be success, info, warning, error.
{{</* /notice */>}}
```

{{< notice success "This is a success type of notice" >}}
success notice.
{{< /notice >}}

{{< notice info "This is a info type of notice" >}}
info notice.
{{< /notice >}}

{{< notice warning "This is a warning type of notice" >}}
warning notice.
{{< /notice >}}

{{< notice error "This is a error type of notice" >}}
error notice.
{{< /notice >}}


### Simple box {#simple-box}

```org
{{</* box */>}}
Plain text
{{</* /box */>}}
```

{{< box >}}
Plain text
{{< /box >}}


### Code in multiple language {#code-in-multiple-language}

```org
{{</* codes java javascript */>}}
  {{</* code */>}}
  #+begin_src java
    System.out.Println("Hello World!");
  #+end_src
  {{</* /code */>}}
  {{</* code */>}}
  #+begin_src javascript
    console.log('Hello World!');
  #+end_src
  {{</* /code */>}}
{{</* /codes */>}}
```

{{< codes java javascript >}}
  {{< code >}}

```java
System.out.Println("Hello World!");
```

{{< /code >}}
{{< code >}}

```javascript
console.log('Hello World!');
```

  {{< /code >}}
{{< /codes >}}


### Tab {#tab}

```org
{{</* tabs Windows MacOS Ubuntu */>}}
  {{</* tab */>}}

  *** Windows section

  #+begin_src javascript
    console.log('Hello World!');
  #+end_src

  {{</* /tab */>}}
  {{</* tab */>}}

  *** MacOS section

  Hello world!
  {{</* /tab */>}}
  {{</* tab */>}}

  *** Ubuntu section

  Great!
  {{</* /tab */>}}
{{</* /tabs */>}}
```

{{< tabs Windows MacOS Ubuntu >}}
  {{< tab >}}


### Windows section {#windows-section}

```javascript
console.log('Hello World!');
```

{{< /tab >}}
{{< tab >}}


### MacOS section {#macos-section}

Hello world!
{{< /tab >}}
{{< tab >}}


### Ubuntu section {#ubuntu-section}

  Great!
  {{< /tab >}}
{{< /tabs >}}


### Expand {#expand}

```org
{{</* expand "Expand me" */>}}
Some Markdown Contents
{{</* /expand */>}}
```

{{< expand "Expand me" >}}
Some Markdown Contents

```go
package main

import "fmt"

func main() {
  fmt.Println("hello sky!")
}
```

{{< /expand >}}


### video {#video}

{{< youtube 2liXzaIIyuE >}}


## References {#references}

-   [ox-hugo official site](https://ox-hugo.scripter.co/)
-   [plantuml official site](https://plantuml.com/)
-   [zzo-docs on shortcodes](https://zzo-docs.vercel.app/zzo/shortcodes/)
